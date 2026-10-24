// Voice call between the browsers in a room (WebRTC "mesh": one direct audio
// connection to every other person in the call).
//
// The coordinator never touches the audio. It only forwards small "signal"
// messages between two tabs so they can find each other:
//   1. A joins: sends call_join; the coordinator broadcasts the call list.
//   2. For each pair, the tab with the SMALLER peer id makes an "offer";
//      the other answers. (A fixed rule, so both sides never offer at once.)
//   3. Both sides trade ICE candidates (possible network routes), and the
//      browsers connect directly - via a TURN relay if the network is strict.
//
// The microphone needs a secure page: https:// or http://localhost.

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import { useRoom } from "./room.jsx";

const SPEAKING_LEVEL = 0.01; // audioLevel above this counts as talking

export function useCall() {
  const { state, send, signalHandler } = useRoom();
  const me = state.myPeerId;
  const [inCall, setInCall] = useState(false);
  const [muted, setMuted] = useState(false);
  const [speaking, setSpeaking] = useState({}); // peer_id -> true while talking
  const [error, setError] = useState("");

  const local = useRef(null); // my microphone stream
  const peers = useRef(new Map()); // peer_id -> { pc, audio, pendingIce }
  const ice = useRef(null); // STUN/TURN config from /api/ice
  const name = useRef("");

  // ---------------------------------------------------- speaking indicator
  // WebRTC already measures loudness (audioLevel, 0..1) for each connection:
  // "inbound-rtp" = the other person, "media-source" = my own microphone.
  useEffect(() => {
    if (!inCall || !me) return;
    const timer = setInterval(async () => {
      const now = {};
      for (const [id, { pc }] of peers.current) {
        try {
          (await pc.getStats()).forEach((r) => {
            if (r.kind !== "audio" || r.audioLevel === undefined) return;
            if (r.type === "inbound-rtp") now[id] = r.audioLevel > SPEAKING_LEVEL;
            if (r.type === "media-source") now[me] = now[me] || (!muted && r.audioLevel > SPEAKING_LEVEL);
          });
        } catch {
          /* connection closing */
        }
      }
      setSpeaking(now);
    }, 300);
    return () => clearInterval(timer);
  }, [inCall, me, muted]);

  // ---------------------------------------------------- peer connections
  const closePeer = (id) => {
    const p = peers.current.get(id);
    if (!p) return;
    p.pc.close();
    p.audio.srcObject = null;
    peers.current.delete(id);
  };

  const getPeer = useCallback(
    (id) => {
      if (peers.current.has(id)) return peers.current.get(id);
      const pc = new RTCPeerConnection(ice.current || {});
      const audio = new Audio();
      audio.autoplay = true;
      const p = { pc, audio, pendingIce: [] };
      peers.current.set(id, p);
      local.current?.getTracks().forEach((t) => pc.addTrack(t, local.current));
      pc.onicecandidate = (e) => e.candidate && send({ type: "signal", to: id, data: { candidate: e.candidate.toJSON() } });
      pc.ontrack = (e) => {
        audio.srcObject = e.streams[0];
        audio.play().catch(() => {});
      };
      return p;
    },
    [send],
  );

  const call = async (id) => {
    const { pc } = getPeer(id);
    await pc.setLocalDescription(await pc.createOffer());
    send({ type: "signal", to: id, data: { sdp: pc.localDescription.toJSON() } });
  };

  // Messages from other tabs (offer / answer / ICE candidate).
  useEffect(() => {
    signalHandler.current = async (from, data) => {
      if (!local.current) return; // not in the call
      const p = getPeer(from);
      try {
        if (data.sdp) {
          await p.pc.setRemoteDescription(data.sdp);
          for (const c of p.pendingIce.splice(0)) await p.pc.addIceCandidate(c);
          if (data.sdp.type === "offer") {
            await p.pc.setLocalDescription(await p.pc.createAnswer());
            send({ type: "signal", to: from, data: { sdp: p.pc.localDescription.toJSON() } });
          }
        } else if (data.candidate) {
          // Candidates can arrive before the offer/answer: keep them until then.
          if (p.pc.remoteDescription) await p.pc.addIceCandidate(data.candidate);
          else p.pendingIce.push(data.candidate);
        }
      } catch (e) {
        console.warn("call signal failed", e);
      }
    };
    return () => {
      signalHandler.current = null;
    };
  }, [getPeer, send, signalHandler]);

  // Whenever the call list changes: connect to newcomers, drop people who left.
  useEffect(() => {
    if (!inCall || !me) return;
    const ids = new Set(state.call.map((p) => p.peer_id));
    for (const id of [...peers.current.keys()]) if (!ids.has(id)) closePeer(id);
    for (const id of ids) {
      if (id !== me && !peers.current.has(id) && me < id) call(id); // smaller id offers
    }
  }, [state.call, inCall, me]); // eslint-disable-line react-hooks/exhaustive-deps

  // Our socket reconnected (new peer id): the coordinator dropped us from the
  // call, so start fresh and join again.
  const lastMe = useRef(me);
  useEffect(() => {
    if (inCall && me && lastMe.current && me !== lastMe.current) {
      [...peers.current.keys()].forEach(closePeer);
      send({ type: "call_join", name: name.current });
    }
    lastMe.current = me;
  }, [me]); // eslint-disable-line react-hooks/exhaustive-deps

  // ------------------------------------------------------------ actions
  const join = async (who) => {
    setError("");
    if (!window.isSecureContext || !navigator.mediaDevices) {
      setError("Voice needs a secure page: open the dashboard via https:// (e.g. the Render URL) or http://localhost.");
      return;
    }
    try {
      ice.current = ice.current || (await api("/api/ice"));
      local.current = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    } catch (e) {
      setError(e.name === "NotAllowedError" ? "Microphone permission was denied." : `Could not start the call: ${e.message}`);
      return;
    }
    name.current = who;
    setMuted(false);
    setInCall(true);
    send({ type: "call_join", name: who });
  };

  const leave = useCallback(() => {
    send({ type: "call_leave" });
    [...peers.current.keys()].forEach(closePeer);
    local.current?.getTracks().forEach((t) => t.stop());
    local.current = null;
    setSpeaking({});
    setInCall(false);
  }, [send]);

  const toggleMute = () => {
    const next = !muted;
    local.current?.getAudioTracks().forEach((t) => (t.enabled = !next));
    setMuted(next);
  };

  // Leaving the room page ends the call.
  const leaveRef = useRef(leave);
  leaveRef.current = leave;
  useEffect(() => () => local.current && leaveRef.current(), []);

  return { inCall, muted, speaking, error, join, leave, toggleMute };
}

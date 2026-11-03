"""Offline streaming/TTS checks; network, models, and audio devices are mocked."""

import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from tts_service import realtime


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("voice_transcribe", ROOT / "voice-transcribe.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class SpeechTests(unittest.TestCase):
    def backend(self):
        engine = MagicMock()
        engine.get_voices.return_value = [types.SimpleNamespace(name="af_heart")]
        stream = MagicMock()
        module = types.SimpleNamespace(
            KokoroEngine=MagicMock(return_value=engine),
            TextToAudioStream=MagicMock(return_value=stream),
        )
        return module, engine, stream

    @staticmethod
    def emit_audio(**kwargs):
        kwargs["on_audio_chunk"](b"audio")

    def test_session_warms_once_and_reuses_model_across_turns(self):
        module, engine, stream = self.backend()
        stream.play.side_effect = self.emit_audio
        with patch.dict(sys.modules, {"RealtimeTTS": module}):
            session = realtime.KokoroSession(speed=1.1)
            session.speak_stream(iter(("First response.",)))
            session.speak_stream(iter(("Second response.",)))
            session.close()

        module.KokoroEngine.assert_called_once_with(voice="af_heart", default_speed=1.1)
        self.assertEqual(stream.feed.call_count, 3)  # one warm-up and two turns
        self.assertEqual(stream.play.call_count, 3)
        engine.shutdown.assert_called_once()
        self.assertTrue(stream.play.call_args.kwargs["fast_sentence_fragment"])
        self.assertEqual(stream.play.call_args.kwargs["minimum_first_fragment_length"], 18)
        self.assertEqual(stream.play.call_args.kwargs["force_first_fragment_after_words"], 12)

    def test_first_audio_latency_is_measured_from_selected_origin(self):
        module, _, stream = self.backend()
        stream.play.side_effect = self.emit_audio
        with patch.dict(sys.modules, {"RealtimeTTS": module}), \
             patch.object(realtime.time, "perf_counter", side_effect=[1.0, 2.0, 3.0, 4.0, 12.5]):
            session = realtime.KokoroSession()
            timing = session.speak_stream(
                iter(("Hello there.",)),
                latency_started=10.0,
                latency_origin="end of speech",
            )
            session.close()
        self.assertEqual(timing.first_audio_seconds, 2.5)

    def test_silent_synthesis_is_reported_after_successful_warmup(self):
        module, engine, stream = self.backend()
        play_count = 0

        def play(**kwargs):
            nonlocal play_count
            play_count += 1
            if play_count == 1:
                self.emit_audio(**kwargs)

        stream.play.side_effect = play
        with patch.dict(sys.modules, {"RealtimeTTS": module}):
            session = realtime.KokoroSession()
            with self.assertRaisesRegex(RuntimeError, "no playable audio"):
                session.speak_stream(iter(("Hello.",)))
            session.close()
        stream.stop.assert_called()
        engine.shutdown.assert_called_once()

    def test_ctrl_c_stops_turn_and_session_can_be_closed(self):
        module, engine, stream = self.backend()
        play_count = 0

        def play(**kwargs):
            nonlocal play_count
            play_count += 1
            if play_count == 1:
                self.emit_audio(**kwargs)
            else:
                raise KeyboardInterrupt

        stream.play.side_effect = play
        with patch.dict(sys.modules, {"RealtimeTTS": module}):
            session = realtime.KokoroSession()
            with self.assertRaises(KeyboardInterrupt):
                session.speak_stream(iter(("Hello.",)))
            session.close()
        stream.stop.assert_called()
        engine.shutdown.assert_called_once()

    def test_missing_dependencies_give_install_instructions(self):
        with patch.dict(sys.modules, {"RealtimeTTS": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements-tts.txt"):
                realtime.KokoroSession()

    def test_bad_voice_releases_engine_without_opening_playback(self):
        module, engine, _ = self.backend()
        with patch.dict(sys.modules, {"RealtimeTTS": module}):
            with self.assertRaisesRegex(RuntimeError, "Unknown Kokoro voice"):
                realtime.KokoroSession(voice="missing_voice")
        module.TextToAudioStream.assert_not_called()
        engine.shutdown.assert_called_once()

    def test_invalid_speed_rejected_before_importing_models(self):
        for speed in [0, -1, float("nan"), float("inf")]:
            with self.subTest(speed=speed):
                with self.assertRaisesRegex(RuntimeError, "greater than zero"):
                    realtime.KokoroSession(speed=speed)


class GeminiStreamingTests(unittest.TestCase):
    class Response:
        def __init__(self, lines):
            self.lines = lines

        def __enter__(self):
            return iter(self.lines)

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def test_sse_text_events_are_yielded_incrementally(self):
        events = [
            {"candidates": [{"content": {"parts": [{"text": "Hello "}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "world."}]}}]},
        ]
        lines = []
        for event in events:
            lines.extend([f"data: {json.dumps(event)}\n".encode(), b"\n"])
        response = self.Response(lines)
        history = [{"role": "user", "parts": [{"text": "Earlier"}]}]

        with patch.object(app.urllib.request, "urlopen", return_value=response) as urlopen:
            chunks = list(
                app.iter_gemini_response("Now", "model-name", "key", 5, history=history)
            )

        self.assertEqual(chunks, ["Hello ", "world."])
        request = urlopen.call_args.args[0]
        self.assertIn(":streamGenerateContent?alt=sse", request.full_url)
        payload = json.loads(request.data)
        self.assertEqual(payload["contents"][-1]["parts"][0]["text"], "Now")
        self.assertEqual(len(payload["contents"]), 2)
        system_text = payload["systemInstruction"]["parts"][0]["text"]
        self.assertIn(app.USER_STYLE_PROFILE, system_text)
        self.assertIn("Should I change this now?", system_text)
        self.assertNotIn("Should I change this now?", json.dumps(payload["contents"]))

    def test_named_styles_keep_profile_and_examples_only_guide_mine(self):
        concise = app.build_system_instruction("concise")
        mine = app.build_system_instruction("mine")

        self.assertIn(app.USER_STYLE_PROFILE, concise)
        self.assertIn("especially concise", concise)
        self.assertNotIn("STYLE EXAMPLES", concise)
        self.assertIn("STYLE EXAMPLES", mine)

    def test_response_style_defaults_to_mine_and_preserves_other_choices(self):
        with patch.object(sys, "argv", ["voice-transcribe.py", "--input", __file__]):
            args = app.parse_args()
        self.assertEqual(args.response_style, "mine")
        for style in ("concise", "conversational", "detailed"):
            with patch.object(
                sys,
                "argv",
                ["voice-transcribe.py", "--input", __file__, "--response-style", style],
            ):
                self.assertEqual(app.parse_args().response_style, style)

    def test_empty_stream_is_an_error(self):
        response = self.Response([b"data: {\"promptFeedback\": {\"blockReason\": \"SAFETY\"}}\n", b"\n"])
        with patch.object(app.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "SAFETY"):
                list(app.iter_gemini_response("Hi", "model", "key", 5))


class RecordingTests(unittest.TestCase):
    def test_recording_stops_after_sustained_silence(self):
        import numpy as np

        frames = 1600
        blocks = [
            np.zeros((frames, 1), dtype=np.float32),
            np.full((frames, 1), 0.1, dtype=np.float32),
            np.full((frames, 1), 0.1, dtype=np.float32),
            np.zeros((frames, 1), dtype=np.float32),
            np.zeros((frames, 1), dtype=np.float32),
        ]

        class FakeInputStream:
            def __init__(self):
                self.read_count = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self, frame_count):
                block = blocks[self.read_count]
                self.read_count += 1
                return block, False

        fake_stream = FakeInputStream()
        sounddevice = types.SimpleNamespace(InputStream=MagicMock(return_value=fake_stream))
        with patch.dict(sys.modules, {"sounddevice": sounddevice}):
            result = app.record_audio(
                5,
                silence_duration=0.2,
                silence_threshold=0.05,
                block_duration=0.1,
            )

        self.assertEqual(fake_stream.read_count, 5)
        self.assertEqual(len(result.audio), frames * 5)
        self.assertIsInstance(result.speech_end_time, float)


class WhisperSessionTests(unittest.TestCase):
    def test_model_loads_once_and_transcribes_multiple_turns(self):
        model = MagicMock()
        info = types.SimpleNamespace(language="en")
        model.transcribe.side_effect = [
            (iter([types.SimpleNamespace(text=" First")]), info),
            (iter([types.SimpleNamespace(text=" Second")]), info),
        ]
        whisper_model = MagicMock(return_value=model)
        module = types.SimpleNamespace(WhisperModel=whisper_model)

        with patch.dict(sys.modules, {"faster_whisper": module}):
            session = app.FasterWhisperSession("small.en", "cpu", "int8")
            self.assertEqual(session.transcribe("one.wav", "en"), "First")
            self.assertEqual(session.transcribe("two.wav", "en"), "Second")

        whisper_model.assert_called_once_with(
            "small.en", device="cpu", compute_type="int8"
        )
        self.assertEqual(model.transcribe.call_count, 2)


class AppTests(unittest.TestCase):
    def run_app(
        self,
        *flags,
        input_file=True,
        stream_chunks=("The ", "response."),
        capture_result=("Hello", None),
    ):
        argv = ["voice-transcribe.py"]
        if input_file:
            argv.extend(["--input", __file__])
        argv.extend(flags)
        with patch.object(sys, "argv", argv), \
             patch.object(app, "ensure_dependencies"), \
             patch.object(app, "FasterWhisperSession") as whisper_type, \
             patch.object(app, "capture_transcript", return_value=capture_result), \
             patch.object(
                 app,
                 "iter_gemini_response",
                 side_effect=lambda *args, **kwargs: iter(stream_chunks),
             ), \
             patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("sys.stdout", new_callable=io.StringIO), \
             patch("sys.stderr", new_callable=io.StringIO):
            self.whisper_type = whisper_type
            return app.main()

    def test_text_only_streams_without_loading_optional_speech(self):
        with patch.object(realtime, "KokoroSession") as session:
            self.assertEqual(self.run_app(), 0)
        session.assert_not_called()

    def test_speak_response_streams_chunks_into_one_warmed_session(self):
        session = MagicMock()

        def consume(chunks, latency_started, latency_origin):
            self.assertEqual("".join(chunks), "The response.")
            return realtime.SpeechTiming(first_audio_seconds=0.5)

        session.speak_stream.side_effect = consume
        with patch.object(realtime, "KokoroSession", return_value=session) as session_type, \
             patch.object(app, "speak_response") as old_speak:
            self.assertEqual(self.run_app("--speak-response"), 0)
        session_type.assert_called_once_with(voice="af_heart", speed=1.0)
        session.speak_stream.assert_called_once()
        session.close.assert_called_once()
        old_speak.assert_not_called()

    def test_existing_mac_engine_is_still_available(self):
        with patch.object(app, "speak_response") as old_speak:
            self.assertEqual(
                self.run_app("--speak-response", "--tts-engine", "voxcpm2"), 0
            )
        old_speak.assert_called_once()
        self.assertEqual(old_speak.call_args.args[0], "The response.")

    def test_kokoro_rejects_cloning_before_recording_or_api_call(self):
        with patch.object(app, "ensure_dependencies") as ensure, \
             patch.object(realtime, "KokoroSession") as session:
            self.assertEqual(
                self.run_app("--speak-response", "--tts-reference-audio", "ref.wav"), 1
            )
        ensure.assert_not_called()
        session.assert_not_called()

    def test_playback_interrupt_returns_shell_interrupt_status_and_closes(self):
        session = MagicMock()
        session.speak_stream.side_effect = KeyboardInterrupt
        with patch.object(realtime, "KokoroSession", return_value=session):
            self.assertEqual(self.run_app("--speak-response"), 130)
        session.close.assert_called_once()

    def test_microphone_turn_uses_end_of_speech_as_audio_latency_origin(self):
        session = MagicMock()

        def consume(chunks, latency_started, latency_origin):
            list(chunks)
            return realtime.SpeechTiming(first_audio_seconds=2.0)

        session.speak_stream.side_effect = consume
        with patch.object(realtime, "KokoroSession", return_value=session):
            self.assertEqual(
                self.run_app(
                    "--speak-response",
                    input_file=False,
                    capture_result=("Hello", 123.0),
                ),
                0,
            )
        self.assertEqual(session.speak_stream.call_args.kwargs["latency_started"], 123.0)
        self.assertEqual(
            session.speak_stream.call_args.kwargs["latency_origin"], "end of speech"
        )

    def test_conversation_reuses_session_for_multiple_turns(self):
        session = MagicMock()

        def consume(chunks, latency_started, latency_origin):
            list(chunks)
            return realtime.SpeechTiming(first_audio_seconds=0.5)

        session.speak_stream.side_effect = consume
        with patch.object(realtime, "KokoroSession", return_value=session) as session_type, \
             patch("builtins.input", side_effect=["", "q"]):
            self.assertEqual(
                self.run_app("--speak-response", "--conversation", input_file=False), 0
            )
        session_type.assert_called_once()
        self.whisper_type.assert_called_once_with("small.en", "cpu", "int8")
        self.assertEqual(session.speak_stream.call_count, 2)
        session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

"""Mensagem de audio na caixa de conversa — o botao de microfone do chat.

Criado em 23/10/2026 a pedido do dono: "adicionar o microfone na conversa,
tipo audio". Clica no microfone, fala, clica de novo: o que ele falou vira
texto e entra no chat como se tivesse sido digitado.

## Quem transcreve

Primeiro a Gemini (`gemini-2.5-flash-lite`), com a mesma chave de
`config/api_keys.json`: e rapida e nao precisa baixar modelo nenhum. Se ela
recusar (cota, rede, chave), cai no Whisper local do `modo_economico.py`, que
roda na CPU e nao gasta nada — mas so se o `faster_whisper` estiver instalado.

## Por que um fluxo de microfone separado

A escuta continua do JARVIS e sempre aberta e decide sozinha quando a frase
acabou. Aqui quem decide e o dono, no clique. Por isso a gravacao abre o seu
proprio `InputStream` e devolve tudo de uma vez quando ele para.
"""
from __future__ import annotations

import io
import json
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

TAXA = 16000
FALA_MAX_S = 120.0          # teto de seguranca: esqueceu gravando, nao enche a RAM
MODELOS_TRANSCRICAO = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

BASE = Path(__file__).resolve().parents[1]


class GravadorDeAudio:
    """Liga com `comecar()`, desliga com `parar()` — que devolve o PCM int16."""

    def __init__(self) -> None:
        self._blocos: list[np.ndarray] = []
        self._stream = None
        self._lock = threading.Lock()

    @property
    def gravando(self) -> bool:
        return self._stream is not None

    def comecar(self, dispositivo=None) -> None:
        if self._stream is not None:
            return
        self._blocos = []
        maximo = int(FALA_MAX_S * TAXA)

        def callback(indata, frames, tinfo, status):
            with self._lock:
                if sum(len(b) for b in self._blocos) < maximo:
                    self._blocos.append(indata.reshape(-1).copy())

        # Dispositivo escolhido na engrenagem; "nao-definido" ou lixo cai no
        # padrao do Windows em vez de quebrar o botao.
        try:
            dispositivo = int(dispositivo) if dispositivo is not None else None
        except (TypeError, ValueError):
            dispositivo = None
        try:
            self._stream = sd.InputStream(samplerate=TAXA, channels=1, dtype="int16",
                                          blocksize=640, callback=callback,
                                          device=dispositivo)
        except Exception:
            self._stream = sd.InputStream(samplerate=TAXA, channels=1, dtype="int16",
                                          blocksize=640, callback=callback)
        self._stream.start()

    def parar(self) -> np.ndarray:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        with self._lock:
            blocos, self._blocos = self._blocos, []
        if not blocos:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(blocos)


def _wav(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TAXA)
        wf.writeframes(pcm.astype(np.int16).tobytes())
    return buf.getvalue()


def _chave_gemini() -> str:
    try:
        dados = json.loads((BASE / "config" / "api_keys.json").read_text(encoding="utf-8"))
        return str(dados.get("gemini_api_key") or "").strip()
    except Exception:
        return ""


def _pela_gemini(pcm: np.ndarray) -> str:
    chave = _chave_gemini()
    if not chave:
        raise RuntimeError("chave Gemini ausente")
    from google import genai  # noqa: PLC0415
    from google.genai import types  # noqa: PLC0415
    cliente = genai.Client(api_key=chave, http_options={"timeout": 30000})
    audio = types.Part.from_bytes(data=_wav(pcm), mime_type="audio/wav")
    pedido = ("Transcreva este audio exatamente como foi falado, em portugues do "
              "Brasil. Devolva SO o texto falado, sem aspas, sem comentario. Se "
              "nao houver fala, devolva vazio.")
    ultimo = None
    for modelo in MODELOS_TRANSCRICAO:
        try:
            r = cliente.models.generate_content(model=modelo, contents=[pedido, audio])
            return (r.text or "").strip()
        except Exception as exc:
            ultimo = exc
    raise ultimo or RuntimeError("nenhum modelo transcreveu")


_whisper = None


def _pelo_whisper(pcm: np.ndarray) -> str:
    global _whisper
    from faster_whisper import WhisperModel  # noqa: PLC0415
    if _whisper is None:
        _whisper = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _ = _whisper.transcribe(pcm.astype(np.float32) / 32768.0, language="pt",
                                  beam_size=1, vad_filter=True,
                                  condition_on_previous_text=False)
    return " ".join(s.text for s in segs).strip()


def transcrever(pcm: np.ndarray) -> str:
    """Texto do que foi falado. Vazio se nao deu para entender nada."""
    if pcm is None or len(pcm) < TAXA * 0.4:      # menos de 0,4 s: clique sem fala
        return ""
    erros = []
    for motor in (_pela_gemini, _pelo_whisper):
        try:
            return motor(pcm)
        except Exception as exc:
            erros.append(f"{motor.__name__}: {str(exc)[:80]}")
    raise RuntimeError(" | ".join(erros))

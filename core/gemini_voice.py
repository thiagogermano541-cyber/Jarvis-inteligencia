from __future__ import annotations

import asyncio
import json
import threading
import re
import unicodedata
import wave
from pathlib import Path

import sounddevice as sd
from google import genai
from google.genai import types

from core.velocidade_fala import acelerar_pcm16


LIVE_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
RECEIVE_SAMPLE_RATE = 24000


class GeminiCharonVoice:
    """A sessao ao vivo. O nome da classe ficou Charon por historia; a VOZ
    agora vem de fora (05/09/2026, quando a engrenagem virou 11 vozes da
    Gemini). `voz` e o nome do timbre no servidor, nao a chave do personagem.
    """

    def __init__(self, voz: str = "Charon") -> None:
        self.voz = str(voz or "Charon")
        config_path = Path(__file__).resolve().parents[1] / "config" / "api_keys.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        api_key = str(data.get("gemini_api_key") or "").strip()
        if not api_key:
            raise RuntimeError("Chave Gemini ausente em config/api_keys.json")
        self.client = genai.Client(api_key=api_key)
        self._persistent_cm = None
        self._persistent_session = None
        self._persistent_lock = None

    def speak(self, text: str, on_audio_start=None, on_audio_end=None) -> None:
        asyncio.run(self._speak(text, on_audio_start=on_audio_start, on_audio_end=on_audio_end))

    def synthesize_wav(self, text: str, out: Path) -> Path:
        return asyncio.run(self._synthesize_wav(text, out))

    def _norm(self, value: str) -> str:
        value = unicodedata.normalize("NFD", value or "")
        value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
        value = re.sub(r"[^a-zA-Z0-9]+", " ", value.lower())
        return re.sub(r"\s+", " ", value).strip()

    def _is_safe_transcript(self, expected: str, actual: str) -> bool:
        if not actual:
            return True
        bad = (
            "nao posso",
            "nao consigo",
            "como assistente",
            "nao tenho capacidade",
            "não posso",
            "não consigo",
        )
        actual_norm = self._norm(actual)
        if any(self._norm(term) in actual_norm for term in bad):
            return False
        return True

    async def _speak(self, text: str, on_audio_start=None, on_audio_end=None) -> None:
        chunks, transcript = await self._generate_audio(text)
        if not self._is_safe_transcript(text, transcript):
            raise RuntimeError(f"transcrição de voz diferente do chat: {transcript}")
        if on_audio_start:
            on_audio_start()
        with sd.RawOutputStream(
            samplerate=RECEIVE_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=1024,
        ) as stream:
            try:
                for chunk in chunks:
                    stream.write(chunk)
            finally:
                if on_audio_end:
                    on_audio_end()

    async def _generate_audio(self, text: str) -> tuple[list[bytes], str]:
        cfg = self._live_config()
        chunks: list[bytes] = []
        transcript_parts: list[str] = []
        async with self.client.aio.live.connect(model=LIVE_MODEL, config=cfg) as session:
            await session.send_client_content(
                turns={"parts": [{"text": f"Leia literalmente, sem alterar nenhuma palavra: {text}"}]},
                turn_complete=True,
            )
            async for response in session.receive():
                data = getattr(response, "data", None)
                if data:
                    chunks.append(data)
                server_content = getattr(response, "server_content", None)
                if server_content:
                    out = getattr(server_content, "output_transcription", None)
                    if out and getattr(out, "text", None):
                        transcript_parts.append(out.text)
                    if getattr(server_content, "turn_complete", False):
                        break
        return chunks, " ".join(transcript_parts).strip()

    def _live_config(self) -> types.LiveConnectConfig:
        # SEM PENSAR ANTES DE FALAR (05/09/2026). Este modelo e de audio NATIVO:
        # ele raciocina antes de abrir a boca, e o proprio SDK avisa que a
        # resposta veio com partes `text` e `thought`. Para CONVERSAR isso e
        # otimo — e o que faz a voz do repositorio soar inteligente. Para LER
        # uma frase que o cerebro ja escreveu, e so espera.
        #
        # Medido em A/B alternado, 2 voltas de 3 frases, com a chave do Diego:
        #     pensando ....... 2,22 s ate o primeiro som
        #     sem pensar ..... 1,60 s   (e 1,20 s nos turnos ja aquecidos)
        # O audio saiu do mesmo tamanho: nao se perdeu fala, so a espera.
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            # FALAR COMO GENTE, NAO COMO QUEM LE EM VOZ ALTA (05/09/2026).
            #
            # A instrucao antiga dizia "voce e apenas o sintetizador de voz" e
            # "leia literalmente" — e era exatamente assim que saia: correto e
            # sem alma. O que faz a voz do repositorio de referencia soar viva e
            # que la o modelo esta CONVERSANDO. Aqui ele nao pode conversar (o
            # texto vem pronto do cerebro), mas pode DIZER em vez de ler.
            #
            # A fidelidade continua exigida palavra por palavra: o que muda e a
            # entrega — respiracao, entonacao de pergunta, a emocao que o
            # proprio texto pede.
            system_instruction=(
                "Você é a VOZ do JARVIS falando com o dono dele. Diga o texto que "
                "chegar, palavra por palavra, sem trocar, tirar nem acrescentar "
                "nada — mas diga como GENTE fala, não como quem lê em voz alta: "
                "ritmo natural, respiração entre as frases, a entonação subindo na "
                "pergunta e caindo no fim da afirmação, e a emoção que o próprio "
                "texto pede. Português do Brasil, voz masculina, confiante, próxima "
                "e inteligente, como um mordomo que conhece o dono há anos. Nunca "
                "responda ao texto, nunca comente, nunca avise que vai ler."
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=getattr(self, "voz", "Charon"))
                )
            ),
        )

    async def _ensure_persistent_session(self):
        if self._persistent_lock is None:
            self._persistent_lock = asyncio.Lock()
        if self._persistent_session is None:
            self._persistent_cm = self.client.aio.live.connect(
                model=LIVE_MODEL,
                config=self._live_config(),
            )
            self._persistent_session = await self._persistent_cm.__aenter__()
        return self._persistent_session

    async def _reset_persistent_session(self) -> None:
        cm = self._persistent_cm
        self._persistent_cm = None
        self._persistent_session = None
        if cm is not None:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass

    async def warm_persistent_session(self) -> None:
        if self._persistent_lock is None:
            self._persistent_lock = asyncio.Lock()
        async with self._persistent_lock:
            await self._ensure_persistent_session()

    async def generate_audio_persistent(self, text: str) -> tuple[list[bytes], str]:
        if self._persistent_lock is None:
            self._persistent_lock = asyncio.Lock()
        async with self._persistent_lock:
            last_error = None
            for _attempt in range(2):
                try:
                    session = await self._ensure_persistent_session()
                    chunks: list[bytes] = []
                    transcript_parts: list[str] = []
                    await session.send_client_content(
                        turns={"parts": [{"text": f"Leia literalmente, sem alterar nenhuma palavra: {text}"}]},
                        turn_complete=True,
                    )
                    async for response in session.receive():
                        data = getattr(response, "data", None)
                        if data:
                            chunks.append(data)
                        server_content = getattr(response, "server_content", None)
                        if server_content:
                            out = getattr(server_content, "output_transcription", None)
                            if out and getattr(out, "text", None):
                                transcript_parts.append(out.text)
                            if getattr(server_content, "turn_complete", False):
                                return chunks, " ".join(transcript_parts).strip()
                    raise RuntimeError("sessão literal encerrou sem turn_complete")
                except Exception as exc:
                    last_error = exc
                    await self._reset_persistent_session()
            raise RuntimeError(f"falha na sessão literal: {last_error}")

    async def stream_audio_persistent(self, text: str, on_chunk) -> str:
        """Entrega PCM conforme chega, mantendo a sessão literal já aquecida."""
        if self._persistent_lock is None:
            self._persistent_lock = asyncio.Lock()
        async with self._persistent_lock:
            last_error = None
            for _attempt in range(2):
                emitted = False
                try:
                    session = await self._ensure_persistent_session()
                    transcript_parts: list[str] = []
                    await session.send_client_content(
                        turns={"parts": [{"text": f"Leia literalmente, sem alterar nenhuma palavra: {text}"}]},
                        turn_complete=True,
                    )
                    async for response in session.receive():
                        data = getattr(response, "data", None)
                        if data:
                            emitted = True
                            on_chunk(data)
                        server_content = getattr(response, "server_content", None)
                        if server_content:
                            out = getattr(server_content, "output_transcription", None)
                            if out and getattr(out, "text", None):
                                transcript_parts.append(out.text)
                            if getattr(server_content, "turn_complete", False):
                                return " ".join(transcript_parts).strip()
                    raise RuntimeError("sessão literal encerrou sem turn_complete")
                except Exception as exc:
                    last_error = exc
                    await self._reset_persistent_session()
                    # Nunca reinicia depois de já tocar áudio, evitando duplicação.
                    if emitted:
                        break
            raise RuntimeError(f"falha no streaming literal: {last_error}")

    _laco_proprio = None
    _lock_laco = threading.Lock()

    def speak_para_fila(self, texto: str, on_chunk) -> None:
        """Sincrono: manda TEXTO, devolve os pedacos de audio pelo `on_chunk`.

        Feito para ser chamado de uma thread do modo local, onde a Gemini serve
        APENAS de boca — o cerebro esta no OpenRouter e as tarefas no Codex.

        Por que isso nao queima a cota como a conversa de manha: a cota do Live
        e contada por SESSAO e por SEGUNDO DE AUDIO ENVIADO. Mandando so texto,
        nao ha audio subindo, e a sessao persistente aguenta horas — medido em
        30/08, com uma conexao servindo varios turnos enquanto os modelos
        cobrados por pedido estavam todos estourados.

        O laco de eventos e proprio e vive numa thread separada: a sessao
        persistente precisa sempre do MESMO laco, e o laco do JARVIS esta
        ocupado com o microfone e a caixa de som.
        """
        with self._lock_laco:
            if GeminiCharonVoice._laco_proprio is None:
                laco = asyncio.new_event_loop()

                def rodar():
                    asyncio.set_event_loop(laco)
                    laco.run_forever()

                threading.Thread(target=rodar, daemon=True,
                                 name="charon-loop").start()
                GeminiCharonVoice._laco_proprio = laco
            laco = GeminiCharonVoice._laco_proprio

        # A sessao persistente entrega o PRIMEIRO turno e depois trava: medido
        # em 30/08, frase 1 em 5,7 s e frases 2 e 3 estourando o prazo. O
        # iterador do `receive()` termina a cada turno completo e reabri-lo
        # dentro da mesma sessao nem sempre volta.
        #
        # Entao: tenta a sessao aberta; se estourar o prazo, DERRUBA a sessao e
        # tenta de novo com uma nova. Cair para uma sessao nova custa alguns
        # segundos; ficar mudo custa a conversa.
        # NUNCA repetir depois de ja ter saido audio: em 30/08 a primeira
        # tentativa emitiu a fala inteira e estourou o prazo no fim; a segunda
        # falou TUDO DE NOVO, e o Diego ouviu a mesma frase duas vezes.
        ja_saiu = {"sim": False}

        def _marcado(raw):
            if raw:
                ja_saiu["sim"] = True
            on_chunk(raw)

        for tentativa in (1, 2):
            futuro = asyncio.run_coroutine_threadsafe(
                self.stream_audio_persistent(texto, _marcado), laco)
            try:
                futuro.result(timeout=20)
                return
            except Exception:
                futuro.cancel()
                if ja_saiu["sim"]:
                    # Ja falou. Estourar o prazo no fim nao justifica repetir.
                    return
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._reset_persistent_session(), laco).result(timeout=8)
                except Exception:
                    pass
                if tentativa == 2:
                    raise

    async def _synthesize_wav(self, text: str, out: Path) -> Path:
        cfg = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=(
                "Fale exatamente o texto recebido, em português do Brasil, "
                "com voz formal, natural e inteligente. Não adicione nada."
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
                )
            ),
        )
        chunks, transcript = await self._generate_audio(text)
        if not self._is_safe_transcript(text, transcript):
            raise RuntimeError(f"transcrição de voz diferente do chat: {transcript}")
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RECEIVE_SAMPLE_RATE)
            wf.writeframes(b"".join(chunks))
        return out

# ---------------------------------------------------------------------------
# Voz Charon pelo modelo de FALA (barato) — 30/08/2026
#
# A GeminiCharonVoice acima abre uma sessao do Live API so para ler uma frase.
# O Live API cobra por SEGUNDO DE AUDIO e carrega prompt + ferramentas em toda
# resposta (~10.116 fichas fixas medidas). Ler "Pronto, chefe." por esse caminho
# custa o mesmo que uma resposta inteira.
#
# O gemini-2.5-flash-preview-tts aceita a MESMA voz Charon, e um modelo de fala
# puro (nao raciocina, nao recusa, nao reescreve) e vive em OUTRO balde de cota
# — medido: 328 fichas na frase de teste, contra a sessao Live inteira.
#
# Por ser outro balde, ele continua falando depois que a cota do ao-vivo acaba.
# E por nao raciocinar, ele nao inventa nem recusa: a conferencia de transcricao
# que a classe de cima precisa fazer aqui nao tem motivo de existir.
#
# Medido em 30/08: 2 a 3,5 s ate o audio ficar pronto, contra ~1 s do ao-vivo.
# O modelo NAO entrega o audio em pedacos (generate_content_stream devolve tudo
# de uma vez), entao nao ha ganho em usar a versao de streaming do SDK.
import re as _re_fala

TTS_MODEL = "gemini-2.5-flash-preview-tts"

# FILA DE MODELOS DE FALA (31/08/2026).
#
# Cada modelo tem o SEU proprio teto diario — o `2.5-flash-tts` estourou em
# 100 pedidos e o JARVIS caiu para a voz local no meio da gravacao do Diego.
# Medido na hora, com o mesmo balde ja esgotado no primeiro:
#     gemini-2.5-flash-preview-tts .... BLOQUEADO (teto 100/dia)
#     gemini-3.1-flash-tts-preview ....  2,7 s   OK
#     gemini-2.5-pro-preview-tts ......  2,8 s   OK
#
# Todos aceitam a MESMA voz Charon, entao o dono nao ouve diferenca. Somando os
# baldes, o dia rende cerca de tres vezes mais antes de cair no Kokoro local.
TTS_MODELOS = [
    "gemini-2.5-flash-preview-tts",   # o mais barato, tenta sempre primeiro
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-pro-preview-tts",
]


class GeminiCharonTTS:
    """Mesma voz Charon, pelo modelo de fala — outro balde de cota, bem mais barato."""

    def __init__(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "api_keys.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        api_key = str(data.get("gemini_api_key") or "").strip()
        if not api_key:
            raise RuntimeError("Chave Gemini ausente em config/api_keys.json")
        self.client = genai.Client(api_key=api_key)

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
                )
            ),
        )

    # Qual modelo respondeu por ultimo. Comeca no primeiro da fila e so anda
    # para a frente quando o balde do dia estoura — assim uma resposta nao paga
    # o pedagio de tentar um modelo que ja se sabe esgotado.
    _modelo_atual = 0

    def _pcm_sync(self, text: str) -> bytes:
        """Chamada bloqueante do SDK. Sempre rodar fora do laco de eventos.

        Percorre a fila de modelos: quando um estoura o teto do dia (429), o
        proximo assume com a MESMA voz Charon. So depois de todos falharem e que
        o JARVIS cai para a voz local.
        """
        ultimo = None
        for passo in range(len(TTS_MODELOS)):
            indice = (GeminiCharonTTS._modelo_atual + passo) % len(TTS_MODELOS)
            modelo = TTS_MODELOS[indice]
            try:
                resp = self.client.models.generate_content(
                    model=modelo,
                    contents=text,
                    config=self._config(),
                )
            except Exception as exc:
                ultimo = exc
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    continue          # balde do dia acabou: proximo da fila
                raise
            for cand in (resp.candidates or []):
                content = getattr(cand, "content", None)
                for part in (getattr(content, "parts", None) or []):
                    inline = getattr(part, "inline_data", None)
                    data = getattr(inline, "data", None)
                    if data:
                        if indice != GeminiCharonTTS._modelo_atual:
                            GeminiCharonTTS._modelo_atual = indice
                            print(f"[JARVIS] voz Charon agora pelo {modelo}")
                        # A CADENCIA (01/09/2026). Este modelo nao tem parametro
                        # de velocidade — pedir pressa dentro do texto faz ele
                        # LER o pedido em voz alta. Entao o audio sai daqui ja
                        # acelerado, sem mudar o tom da voz. Ver
                        # `core/velocidade_fala.py`.
                        return acelerar_pcm16(data)
            ultimo = RuntimeError(f"{modelo} nao devolveu audio")
        raise ultimo or RuntimeError("nenhum modelo de fala respondeu")

    async def pcm(self, text: str) -> bytes:
        """PCM 16 bits, mono, 24 kHz — mesmo formato que a fila de audio ja espera."""
        text = str(text or "").strip()
        if not text:
            return b""
        # to_thread e obrigatorio: o SDK bloqueia, e travar o laco aqui
        # engasgaria o microfone e o alto-falante junto.
        return await asyncio.to_thread(self._pcm_sync, text)

    async def stream_audio(self, text: str, on_chunk) -> str:
        """Mesma assinatura da versao Live, para trocar uma pela outra sem cirurgia.

        Devolve o proprio texto como transcricao: modelo de fala le literalmente,
        entao nao existe divergencia entre o que foi escrito e o que foi falado.
        """
        raw = await self.pcm(text)
        if not raw:
            raise RuntimeError("modelo de fala devolveu audio vazio")
        for offset in range(0, len(raw), 2400):
            on_chunk(raw[offset: offset + 2400])
        return text

    # Quebra a resposta em sentencas. Frase curta demais nao vale um pedido
    # separado (o custo fixo de ida e volta come o ganho), entao pedacos de
    # menos de 25 caracteres sao emendados no anterior.
    _FIM_DE_FRASE = _re_fala.compile(r"(?<=[.!?…:;])\s+")

    @classmethod
    def _sentencas(cls, texto: str, minimo: int = 25) -> list:
        """No MAXIMO dois pedidos: a primeira frase e todo o resto.

        Por que dois e nao um por frase: a chave do Diego tem teto de **100
        pedidos por dia** neste modelo de fala (medido em 31/08/2026, quando
        estourou e o JARVIS caiu para a voz local). Um pedido por frase gastava
        3 e 4 fichas por resposta e queimou o dia inteiro numa noite.

        Dois resolve os dois lados: o primeiro sai rapido (a frase e curta) e
        toca enquanto o segundo e gerado, e a conta do dia rende o triplo.
        Resposta curta continua sendo UM pedido so.
        """
        texto = str(texto or "").strip()
        if not texto:
            return []
        # Resposta curta nao vale dividir: uma frase so ja sai rapido.
        if len(texto) <= 140:
            return [texto]
        partes = [p.strip() for p in cls._FIM_DE_FRASE.split(texto) if p.strip()]
        if len(partes) < 2:
            return [texto]
        primeira = partes[0]
        # Primeira frase curta demais nao ganha nada: junta a seguinte.
        i = 1
        while len(primeira) < minimo and i < len(partes):
            primeira += " " + partes[i]
            i += 1
        resto = " ".join(partes[i:]).strip()
        return [primeira, resto] if resto else [primeira]

    def speak_para_fila(self, texto: str, on_chunk) -> None:
        """Mesmo contrato da versao Live: manda texto, devolve audio em pedacos.

        Sincrono de proposito — quem chama e uma thread do modo local, e o laco
        de eventos do JARVIS esta ocupado com microfone e caixa de som.

        FRASE POR FRASE (30/08/2026). Gerar a resposta inteira de uma vez
        custava de 4 a 9 s ate sair o PRIMEIRO som — foi a queixa do Diego de
        "dezesseis, vinte segundos pra responder". Gerando por sentenca, a
        primeira sai em 1,5 a 2,5 s e as outras nascem enquanto ela toca.
        """
        texto = str(texto or "").strip()
        if not texto:
            return
        frases = self._sentencas(texto) or [texto]
        saiu_algum = False
        for frase in frases:
            try:
                raw = self._pcm_sync(frase)
            except Exception:
                # Uma frase que falha no meio nao pode calar o resto da
                # resposta; o que ja saiu continua valendo.
                if saiu_algum:
                    break
                raise
            if not raw:
                continue
            saiu_algum = True
            for offset in range(0, len(raw), 2400):
                on_chunk(raw[offset: offset + 2400])
        if not saiu_algum:
            raise RuntimeError("modelo de fala devolveu audio vazio")

    async def stream_audio_persistent(self, text: str, on_chunk) -> str:
        """Nome que o main.py usa na confirmacao literal. Aqui nao ha sessao
        persistente: cada fala e um pedido isolado, que e justamente o barato."""
        return await self.stream_audio(text, on_chunk)

    async def warm_persistent_session(self) -> None:
        """Nao ha o que aquecer: sem sessao aberta, nao ha sessao para cair.
        Existe so para o main.py poder trocar uma classe pela outra sem cirurgia."""
        return None

    def speak(self, text: str) -> None:
        """Fala e espera terminar. Para avisos soltos, fora da fila de audio."""
        raw = asyncio.run(self.pcm(text))
        if not raw:
            return
        with sd.RawOutputStream(
            samplerate=RECEIVE_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=1024,
        ) as stream:
            stream.write(raw)

    def synthesize_wav(self, text: str, out: Path) -> Path:
        raw = asyncio.run(self.pcm(text))
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RECEIVE_SAMPLE_RATE)
            wf.writeframes(raw)
        return out


# ---------------------------------------------------------------------------
# A VOZ CHARON PELO CAMINHO QUE DURA O DIA INTEIRO NO GRATUITO — 05/09/2026
#
# O Diego trouxe o repositorio de referencia (Mark-LII) dizendo que a voz DE LA
# nao acaba. Fui ver: ele fala com o MESMO modelo e a MESMA voz que a
# `GeminiCharonVoice` la de cima — `gemini-2.5-flash-native-audio-preview`, pelo
# **Live API**. Nao ha segredo nenhum no repositorio; o que mudou foi AQUI.
#
# Em 30/08/2026, com a chave PAGA na mao, o `main.py` foi trocado para o modelo
# de fala por pedido (`gemini-2.5-flash-preview-tts`) porque ele custa menos
# FICHAS por frase. Com chave gratuita a conta se inverte, e essa e a regra que
# explica tudo:
#
#     Live API .............. cobrado por SESSAO  -> uma conexao serve o dia
#     modelo de fala ........ cobrado por PEDIDO  -> 10 por dia no gratuito
#
# Comparar ficha com ficha nao vale quando as duas cotas nao sao contadas na
# mesma moeda. Entao a ordem volta a ser a do repositorio: **ao-vivo primeiro**.
#
# O modelo de fala NAO foi jogado fora — ele vive em outro balde de cota, entao
# continua servindo depois que o ao-vivo recusa. Vira reserva, nao caminho.
# ---------------------------------------------------------------------------

class VozCharon:
    """Charon: tenta o ao-vivo (gratis o dia todo) e cai no modelo de fala.

    Tem o mesmo contrato das duas classes acima — `speak_para_fila`,
    `stream_audio_persistent` e `warm_persistent_session` —, entao o `main.py`
    troca uma pela outra sem cirurgia.
    """

    # Ate quando nao vale a pena tentar o ao-vivo de novo. Insistir a cada frase
    # so faria o dono esperar o erro chegar, de novo e de novo.
    _vivo_ate = 0.0

    def __init__(self, voz: str = "Charon") -> None:
        self.voz = str(voz or "Charon")
        self._vivo = None
        self._barato = None

    # -- os dois motores, criados so quando precisam --------------------------

    def _ao_vivo(self):
        # TROCOU DE VOZ, TROCA A SESSAO. A voz vai na configuracao da conexao:
        # reaproveitar a sessao antiga faria o cliente escolher a Leda na
        # engrenagem e continuar ouvindo a anterior ate reabrir o programa.
        if self._vivo is None or getattr(self._vivo, "voz", None) != self.voz:
            self._vivo = GeminiCharonVoice(self.voz)
        return self._vivo

    def _de_reserva(self):
        if self._barato is None:
            self._barato = GeminiCharonTTS()
        return self._barato

    def _anotar_falha(self, exc: Exception) -> None:
        """Quanto tempo calar o ao-vivo depende do QUE ele reclamou."""
        import time as _t  # noqa: PLC0415
        texto = str(exc)
        baixo = texto.lower()
        cota = ("429" in texto or "RESOURCE_EXHAUSTED" in texto
                or "quota" in baixo)
        # CHAVE RECUSADA NAO E FALHA PASSAGEIRA. Sem esta faixa, uma chave
        # invalida fazia o ao-vivo ser tentado DE NOVO a cada minuto, e cada
        # tentativa e uma ida a rede com o dono esperando o erro chegar. Fica
        # de lado ate a proxima abertura do programa (a chave nova entra por
        # ali, na tela de chaves).
        # "ausente" cobre o cliente que instalou e nunca pos chave nenhuma: com
        # a Charon virando a voz PADRAO, sem esta linha ele tentaria o ao-vivo em
        # toda frase, para sempre, so para ouvir o mesmo erro.
        chave_ruim = ("api key not valid" in baixo or "api_key_invalid" in baixo
                      or "api key expired" in baixo or "chave gemini ausente" in baixo)
        # QUANTO CALAR DEPENDE DO QUE ESTOUROU (05/09/2026).
        #
        # Estava fixo em 15 min para qualquer 429 — e o 429 que a gente ve na
        # pratica e o de RITMO (limite por minuto), que passa em segundos. O
        # Diego levou 15 minutos na voz de reserva por causa de um aperto que
        # ja tinha passado quando ele falou a frase seguinte.
        #
        # O corpo do erro traz `retryDelay` quando o proprio Google sabe o
        # prazo; `_quanto_calar_a_gemini` (no cerebro) ja sabe ler isso e
        # separar "cota do DIA" de "limite por MINUTO". Reaproveitar e melhor do
        # que ter duas regras diferentes para a mesma familia de erro.
        if chave_ruim:
            espera = 12 * 3600.0
        elif cota:
            try:
                from core.cerebro_openrouter import _quanto_calar_a_gemini  # noqa: PLC0415
                espera = float(_quanto_calar_a_gemini(texto)[0])
            except Exception:
                espera = 60.0
        else:
            espera = 60.0
        VozCharon._vivo_ate = _t.time() + espera
        try:
            from main import _voice_diag  # noqa: PLC0415
            _motivo = "chave recusada" if chave_ruim else ("cota" if cota else "erro")
            _voice_diag("charon_ao_vivo_fora",
                        f"{_motivo}: {texto[:90]} "
                        + ("(so na proxima abertura, com chave nova)" if chave_ruim
                           else f"(reserva por {espera/60:.0f} min)"))
        except Exception:
            pass

    # -- o contrato que o main.py usa ----------------------------------------

    def speak_para_fila(self, texto: str, on_chunk) -> None:
        import time as _t  # noqa: PLC0415
        # NUNCA REPETIR DEPOIS DE JA TER SAIDO AUDIO. Se o ao-vivo falou a frase
        # inteira e so estourou o prazo no fim, cair para a reserva faria o dono
        # ouvir tudo DUAS vezes — armadilha que ja pegou este projeto em 30/08.
        saiu = {"sim": False}

        def _marcado(raw):
            if raw:
                saiu["sim"] = True
            on_chunk(raw)

        if _t.time() >= VozCharon._vivo_ate:
            try:
                self._ao_vivo().speak_para_fila(texto, _marcado)
                return
            except Exception as exc:
                if saiu["sim"]:
                    return
                self._anotar_falha(exc)
        self._de_reserva().speak_para_fila(texto, on_chunk)

    async def stream_audio_persistent(self, text: str, on_chunk) -> str:
        import time as _t  # noqa: PLC0415
        saiu = {"sim": False}

        def _marcado(raw):
            if raw:
                saiu["sim"] = True
            on_chunk(raw)

        if _t.time() >= VozCharon._vivo_ate:
            try:
                return await self._ao_vivo().stream_audio_persistent(text, _marcado)
            except Exception as exc:
                if saiu["sim"]:
                    return ""
                self._anotar_falha(exc)
        return await self._de_reserva().stream_audio_persistent(text, on_chunk)

    async def warm_persistent_session(self) -> None:
        """Abre a sessao do ao-vivo antes da primeira fala.

        Falha aqui NAO derruba nada nem cala o ao-vivo: a sessao pode subir na
        hora da fala. Aquecer e cortesia, nao requisito.
        """
        import time as _t  # noqa: PLC0415
        if _t.time() < VozCharon._vivo_ate:
            return None
        try:
            return await self._ao_vivo().warm_persistent_session()
        except Exception as exc:
            try:
                from main import _voice_diag  # noqa: PLC0415
                _voice_diag("charon_nao_aqueceu", str(exc)[:110])
            except Exception:
                pass
            return None

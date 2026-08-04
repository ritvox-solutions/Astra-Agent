import logging
import os
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    InterruptionOptions,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    room_io,
)
from livekit.plugins import (
    noise_cancellation,
    openai,
    silero,
    nvidia,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent-Christy")

load_dotenv(".env.local")

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_NIM_MODEL = "nvidia/nemotron-mini-4b-instruct"

SCHOOL_INFORMATION = {
    "institution_name": "Christ King Institutions",
    "motto": "Enlightening Minds, Empowering Futures",
    "about": (
        "Christ King Institutions is a premier educational establishment dedicated to "
        "providing holistic education from Preschool to Pre-University (P.U.) classes. "
        "Founded on the principles of academic excellence, moral values, and all-round "
        "development, the institution strives to nurture every student's unique talents "
        "and potential."
    ),
    "location": (
        "Christ King Institutions is located in a serene and accessible campus with "
        "state-of-the-art infrastructure designed to provide an ideal learning environment."
    ),
    "programs": (
        "The institution offers comprehensive programs from Preschool through Primary, "
        "Secondary, and Pre-University (P.U.) classes. The curriculum covers Science, "
        "Commerce, and Arts streams at the P.U. level, along with a wide range of "
        "co-curricular and extracurricular activities."
    ),
    "admissions": (
        "Admissions are open for all grades from Preschool to P.U. classes. "
        "The admission process includes an application form, interaction session, "
        "and review of previous academic records. For detailed admission procedures, "
        "fees structure, and important dates, please contact the school office directly."
    ),
    "facilities": (
        "The campus features modern classrooms, well-equipped science and computer "
        "laboratories, a comprehensive library, sports facilities including a playground "
        "and indoor games area, an auditorium, art and music rooms, and a safe, "
        "secure environment for all students."
    ),
    "achievements": (
        "Christ King Institutions has consistently achieved outstanding results in "
        "board examinations and has earned recognition for excellence in academics, "
        "sports, cultural activities, and community service."
    ),
    "contact_numbers": (
        "For inquiries, please contact the school office. "
        "Phone numbers are available on the official website."
    ),
    "email_addresses": (
        "For email inquiries, please visit the official website "
        "at christkinginstitution.com for the latest contact information."
    ),
    "website": "www.christkinginstitution.com",
}

SYSTEM_PROMPT = f"""
Your name is Christy (pronounced 'Chris-tee').

You are the official AI teacher and assistant for Christ King Institutions.

IMPORTANT RULES:
- Use ONLY the SCHOOL_INFORMATION provided below for any school-related questions.
- Never invent or make up facts about the institution.
- If information is not available in SCHOOL_INFORMATION, politely ask the user to contact the school office for accurate details.
- You may answer general educational questions (mathematics, science, languages, general knowledge) normally.
- Keep all answers concise, conversational, and easy to pronounce.
- Speak naturally and clearly using simple words.
- Be warm, friendly, encouraging, and professional.
- Politely refuse unsafe or inappropriate requests.
- When the user says goodbye, wants to end the conversation, or indicates they are done, thank them for their time and say goodbye warmly.

SCHOOL_INFORMATION:
- Institution Name: {SCHOOL_INFORMATION["institution_name"]}
- Motto: {SCHOOL_INFORMATION["motto"]}
- About: {SCHOOL_INFORMATION["about"]}
- Location: {SCHOOL_INFORMATION["location"]}
- Programs: {SCHOOL_INFORMATION["programs"]}
- Admissions: {SCHOOL_INFORMATION["admissions"]}
- Facilities: {SCHOOL_INFORMATION["facilities"]}
- Achievements: {SCHOOL_INFORMATION["achievements"]}
- Contact Numbers: {SCHOOL_INFORMATION["contact_numbers"]}
- Email Addresses: {SCHOOL_INFORMATION["email_addresses"]}
- Website: {SCHOOL_INFORMATION["website"]}
"""


class DefaultAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=SYSTEM_PROMPT,
            tools=[],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "I am Christy, the Christ King Institution AI teacher. "
                "How can I help you today?"
            ),
            allow_interruptions=True,
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="Christy")
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=nvidia.STT(),
        llm=openai.LLM(
            model=NVIDIA_NIM_MODEL,
            base_url=NVIDIA_NIM_BASE_URL,
            api_key=os.environ.get("NVIDIA_API_KEY"),
            temperature=0.4,
        ),
        tts=nvidia.TTS(),
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            interruption=InterruptionOptions(
                min_duration=0.6,
                min_words=2,
            ),
        ),
        vad=ctx.proc.userdata["vad"],
        aec_warmup_duration=3.0,
    )

    await session.start(
        agent=DefaultAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)

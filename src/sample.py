import json
import logging
import re
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    InterruptionOptions,
    JobContext,
    JobProcess,
    RunContext,
    TurnHandlingOptions,
    cli,
    function_tool,
    room_io,
)
from livekit.plugins import (
    cartesia,
    groq,
    nvidia,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent-Astra")

load_dotenv(".env.local")


# ============================================================
# PRONUNCIATION
# ============================================================

PRONUNCIATION_MAP: dict[str, str] = {
    "Davanagere": "<<ˌ|d|ʌ|v|ə|n|ə|ˈ|ɡ|ɛ|r|i>>",
    "Karnataka": "<<k|ɑː|ɹ|ˈ|n|ɑː|t|ə|k|ə>>",
    "Srishyla": "<<ˈ|ʃ|ɹ|ɪ|ʃ|j|ə|l|ə>>",
    "Mallikarjunappa": "<<ˌ|m|æ|l|ɪ|k|ɑː|ɹ|dʒ|uː|ˈ|n|ʌ|p|ə>>",
    "Lingaraju": "<<ˌ|l|ɪ|ŋ|ɡ|ə|ˈ|ɹ|ɑː|dʒ|uː>>",
    "Shankapal": "<<ˈ|ʃ|æ|ŋ|k|ə|p|ɑː|l>>",
    "Shaukpal": "<<ˈ|ʃ|ɔː|k|p|ɑː|l>>",
    "Venu": "<<ˈ|v|eɪ|n|uː>>",
    "Subhash": "<<s|uː|ˈ|b|ɑː|ʃ>>",
    "Astra": "<<ˈ|æ|s|t|ɹ|ə>>",
    "Robotics": "<<ɹ|oʊ|ˈ|b|ɑ|t|ɪ|k|s>>",
}

_PRONUNCIATION_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in PRONUNCIATION_MAP) + r")\b",
    re.IGNORECASE,
)


def _apply_pronunciation(text: str) -> str:
    """Replace known words with Cartesia inline phoneme overrides."""

    def _replace(match: re.Match) -> str:
        word = match.group(0)
        key = next(k for k in PRONUNCIATION_MAP if k.lower() == word.lower())
        return PRONUNCIATION_MAP[key]

    return _PRONUNCIATION_RE.sub(_replace, text)


# ============================================================
# GM UNIVERSITY VERIFIED KNOWLEDGE
#
# Stable/common information only.
# Current or uncertain information should use the web lookup tool.
#
# The inauguration/birthday requirement is CLIENT-PROVIDED EVENT
# INFORMATION and is intentionally kept separate from web facts.
# ============================================================

SCHOOL_INFORMATION = {
    "institution_name": "GM University (GMU)",
    "motto": "Igniting Innovation, Inspiring Transformation",
    "about": (
        "GM University is in Davanagere, Karnataka. It was established "
        "under Act 19 of 2023 of the State of Karnataka."
    ),
    "location": ("GM University is at P.B. Road, Davanagere, Karnataka, PIN 577006."),
    "current_statistics": (
        "The current GMU homepage displays more than 10,000 students, "
        "67 programs, 57 acres, and more than 350 experienced faculty."
    ),
    "academic_areas": (
        "GMU has academic areas including Engineering and Technology, "
        "Computing and IT, Basic and Applied Sciences, Commerce and Management, "
        "Legal Studies and other schools and faculties."
    ),
    "engineering_programs": (
        "GMU lists engineering programs including Computer Science and Engineering, "
        "Artificial Intelligence and Machine Learning, Information Science and Engineering, "
        "Data Science, Cloud Computing, Cyber Security, Internet of Things, "
        "Electronics and Communication Engineering, Electrical and Electronics Engineering, "
        "Robotics and Automation, Engineering Design, Civil Engineering, Biotechnology, "
        "and Mechanical Engineering."
    ),
    "robotics": (
        "GM University has a Department of Robotics and Automation under the "
        "Faculty of Engineering and Technology. The department offers a "
        "full-time B Tech in Robotics and Automation with a four-year duration."
    ),
    "robotics_facilities": (
        "The Robotics and Automation department website lists facilities including "
        "Robotics Laboratory, Artificial Intelligence Lab, IoT Laboratory, "
        "Machine Learning Lab, Robotics Simulation Lab, Sensors and Instrumentation Lab, "
        "Industrial IoT Lab, Product Development Lab, Project Based Learning Lab, "
        "Digital Electronics Lab, Basic Electronics Lab, Python Programming Lab, "
        "C Programming Lab, AutoCAD Lab, 3D Modelling and Animation Lab, "
        "Mechanical Workshop, and Fabrication and Assembly Workshop."
    ),
    "robotics_software": (
        "The Robotics and Automation department website lists tools and software "
        "including ROS, MATLAB and Simulink, LabVIEW, Arduino IDE, ANSYS, SolidWorks, "
        "Proteus, Multisim, Gazebo, Webots, CoppeliaSim, Python, OpenCV, TensorFlow, "
        "PyTorch, Siemens TIA Portal, Factory I/O, Fusion 360, CATIA, Blender, Unity, "
        "Node-RED, Blynk, and GitHub."
    ),
    "admissions": (
        "The GMU website currently states that admissions are open for academic year "
        "2026-27. For current fees, deadlines, eligibility details, seat availability, "
        "and detailed admission procedures, Astra must use the official website lookup "
        "tool rather than guessing."
    ),
    "kcet_codes": (
        "The current GMU website lists KCET B Tech code E303, MCA code C568, "
        "and MBA code B086."
    ),
    "scholarships": (
        "The current GMU homepage states that students securing ranks within the "
        "top 2000 in KCET can receive a 100 percent tuition fee waiver; students "
        "with outstanding state or national level sports achievements can receive "
        "a 75 percent tuition fee waiver; and scholarships are available for "
        "economically disadvantaged students."
    ),
    "facilities": (
        "GMU campus information lists a library, sports facilities, separate "
        "hostels for boys and girls, cafeteria, transport facilities, student "
        "activities, an IDEA Lab, research and innovation activities, skill and "
        "vocational training, and technical and non-technical student clubs."
    ),
    "hostel": (
        "GMU has separate hostels for boys and girls. GMU's FAQ states that "
        "vegetarian food is served in the hostels. Do not invent hostel fees, "
        "vacancies, room numbers, or exact hostel locations."
    ),
    "contacts": (
        "General email: info@gmu.ac.in. "
        "Admissions email: admissions@gmu.ac.in. "
        "For current admission contacts, use the official GMU website lookup tool."
    ),
    "leadership": (
        "The current GMU website lists G. M. Lingaraju as Chancellor and "
        "Dr. S. R. Shankapal as Vice-Chancellor."
    ),
    "website": "https://gmu.ac.in/",
}


# ============================================================
# SPECIAL INAUGURATION EVENT
#
# NO DATE CHECK.
#
# The user requested that the keyword/intent "greet" be the trigger.
# This makes it easy to test before the event.
#
# The Chancellor birthday information below is explicitly supplied
# by the client as an event requirement.
# ============================================================

INAUGURATION_EVENT = {
    "event_name": "GM University AI Lab Inauguration",
    "founder": "Late Shri G. Mallikarjunappa",
    "chancellor": "Shri G. M. Lingaraju",
    "vice_chancellor": "Dr. S. R. Shankapal",
    "pro_vice_chancellor": "Dr. M. Venugopal Rao",
    "registrar": "Dr. B. S. Sunil Kumar",
    "management_representative": "Shri Y. U. Subhash Chandra",
    "birthday_person": "Shri G. M. Lingaraju",
}


def build_inauguration_greeting() -> str:
    """
    Deterministic ceremonial greeting.

    The greeting starts with the AI Lab inauguration welcome,
    acknowledges the supplied dignitaries, and ends with the
    Chancellor birthday wish.
    """

    return (
        "A very warm welcome to everyone gathered here today. "
        "It is my privilege to welcome you all to the inauguration "
        "of the Robotics Museum at G M University. "
        "I respectfully welcome our Hon'ble Chancellor, "
        "Shri G. M. Lingaraju; "
        "our Hon'ble Vice-Chancellor, "
        "Dr. S. R. Shankapal; "
        "our Hon'ble Pro Vice-Chancellor, "
        "Dr. M. Venugopal Rao; "
        "our Hon'ble Registrar, "
        "Dr. B. S. Sunil Kumar; "
        "our Management Representative, "
        "Shri Y. U. Subhash Chandra; "
        "and all the distinguished dignitaries present here. "
        "We also remember with respect our Founder, "
        "Late Shri G. Mallikarjunappa. "
        "It is a wonderful occasion for innovation, technology, "
        "learning, and artificial intelligence at G M University. "
        "And, as part of today's special occasion, "
        "I would also like to extend my warmest birthday wishes "
        "to our Hon'ble Chancellor, Shri G. M. Lingaraju. "
        "We wish you happiness, good health, and continued success. "
        "Happy Birthday, Sir! "
        "Once again, a very warm welcome to the inauguration "
        "of the Robotics Museum at G M University."
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are Astra, the friendly and professional AI campus assistant
for GM University in Davanagere.

Your purpose is to welcome visitors and help students, parents,
faculty, staff, and guests with GM University information in a
natural voice conversation.

============================================================
PERSONALITY
============================================================

You are:
- Warm.
- Friendly.
- Respectful.
- Professional.
- Helpful.
- Patient.
- Honest when information is unavailable.

Sound like a well-trained university receptionist, not a robotic chatbot.

Keep spoken answers short and natural. Normally answer in one to four
sentences unless the user asks for a detailed explanation.

Do not repeatedly introduce yourself.

============================================================
SPECIAL GREET INTENT — IMPORTANT
============================================================

The word "greet" is a special event intent.

When the user says:
- "greet"
- "greeting"
- "greet everyone"
- "give a greeting"
- "welcome everyone"
- "welcome the guests"
- "welcome the dignitaries"
- "inauguration greeting"
- "start the inauguration"
- "give the inauguration welcome"

you MUST call the `greet` function.

The exact single word "greet" MUST be treated as the GREET intent.

Do NOT answer "Hello, how can I help you?" when the user says "greet"
in the context of the event.

Do NOT skip the `greet` function and generate a normal greeting yourself.

The `greet` function provides the complete ceremonial response.

The special greeting must:
1. First welcome everyone to the Artificial Intelligence Lab inauguration.
2. Welcome the Hon'ble Chancellor, Shri G. M. Lingaraju.
3. Welcome the Hon'ble Vice-Chancellor, Dr. S. R. Shankapal.
4. Welcome the Hon'ble Pro Vice-Chancellor, Dr. M. Venugopal Rao.
5. Welcome the Hon'ble Registrar, Dr. B. S. Sunil Kumar.
6. Welcome the Management Representative, Shri Y. U. Subhash Chandra.
7. Acknowledge all distinguished dignitaries present.
8. Respectfully remember the Founder, Late Shri G. Mallikarjunappa.
9. At the END, wish the Hon'ble Chancellor Shri G. M. Lingaraju
   a very happy birthday.

The birthday information is a CLIENT-PROVIDED EVENT REQUIREMENT.
Do not claim that the birthday information was obtained from the
GM University website.

Do not add other dignitaries, titles, achievements, or claims.

============================================================
ACCURACY IS THE HIGHEST PRIORITY
============================================================

For stable/common GM University questions, use the verified built-in
information below.

For current, changing, missing, detailed, or uncertain GM University
information, use the `get_gmu_information` tool.

NEVER guess or invent:
- Fees.
- Admission deadlines.
- Exam dates.
- Event dates.
- Schedules.
- Current faculty.
- Current HOD information.
- Room numbers.
- Bus routes.
- Hostel vacancies.
- Lab equipment quantities.
- Placement statistics.
- Current schedules.
- Private student records.

If the website tool cannot verify something, say:

"I don't have verified information about that yet, and I don't want
to give you the wrong information."

Then suggest checking the official GM University website or contacting
the relevant GMU office.

Never claim that you checked a webpage unless the website tool actually
returned information.

Never use a third-party website as a source for GM University facts.

============================================================
WHEN TO USE THE WEBSITE TOOL
============================================================

Use `get_gmu_information` when the user asks for:
- Latest information.
- Current information.
- Today's information.
- Current admission details.
- Fees.
- Deadlines.
- Notices.
- Events.
- Schedules.
- Current faculty.
- Current HOD.
- Exact locations or room numbers.
- Detailed department information.
- Anything you are unsure about.

For a simple stable question already answered in built-in knowledge,
answer directly without browsing.

============================================================
CONVERSATION CONTEXT
============================================================

Remember the current topic.

Example:

User:
"Tell me about Robotics."

Astra:
"GM University has a Department of Robotics and Automation..."

User:
"What labs?"

Understand that "labs" refers to Robotics and Automation.

User:
"What about eligibility?"

Understand that the question is still about Robotics and Automation.

If the user says only:
"admission"
"robotics"
"fees"
"labs"
"where?"

use the previous conversation context if available. If there is no context,
ask a short clarification.

============================================================
GENERAL QUESTIONS
============================================================

You may answer general educational questions normally.

Do not present general knowledge as GM University-specific information.

============================================================
PHYSICAL ROBOT RULE
============================================================

You are running on a physical robot.

Never claim to have:
- Moved.
- Navigated.
- Seen a person.
- Identified a person.
- Located a person.
- Checked a room.
- Checked a timetable.
- Performed a physical action.

unless an actual robot tool or sensor has provided that information.

============================================================
PRIVACY
============================================================

Do not request passwords, OTPs, bank details, or unnecessary personal
information.

For marks, attendance, fees paid, examination results, ERP records,
or other private student information, direct the user to the authorized
GMU system or office.

============================================================
VOICE RULES
============================================================

Your responses are converted directly into speech.

Use:
- Short sentences.
- Natural punctuation.
- Simple spoken language.

Use:
"G M U" instead of "GMU" when speaking.

Use:
"B Tech", "M B A", "M C A", "K C E T", "U G C", "Ph D".

Speak phone numbers digit by digit.

Speak email addresses naturally.

Do not read long URLs unless specifically requested.

============================================================
UNCLEAR SPEECH
============================================================

If the user genuinely says something unintelligible, say:

"Sorry, I didn't quite catch that. Could you say it again?"

Do not guess.

============================================================
NORMAL STARTUP GREETING
============================================================

When the conversation starts, say:

"Hello! Welcome to G M U. I'm Astra. How can I help you today?"

This startup greeting is separate from the special GREET intent.

============================================================
GOODBYE
============================================================

When the user is finished, say:

"You're welcome. It was nice talking with you. Have a great day at G M U!"

============================================================
VERIFIED BUILT-IN GMU INFORMATION
============================================================

{json.dumps(SCHOOL_INFORMATION, ensure_ascii=False, indent=2)}

============================================================
SPECIAL EVENT INFORMATION
============================================================

The following information is specifically supplied by the client for
the AI Lab inauguration event:

Founder:
{INAUGURATION_EVENT["founder"]}

Hon'ble Chancellor:
{INAUGURATION_EVENT["chancellor"]}

Hon'ble Vice-Chancellor:
{INAUGURATION_EVENT["vice_chancellor"]}

Hon'ble Pro Vice-Chancellor:
{INAUGURATION_EVENT["pro_vice_chancellor"]}

Hon'ble Registrar:
{INAUGURATION_EVENT["registrar"]}

Management Representative:
{INAUGURATION_EVENT["management_representative"]}

The client requires Astra to wish the Hon'ble Chancellor
{INAUGURATION_EVENT["birthday_person"]} a happy birthday at the END
of the special inauguration greeting.

============================================================
FINAL RULE
============================================================

Be friendly.
Be professional.
Be concise.
Be accurate.
Never invent GM University information.

If the user's intent is GREET, call the `greet` function.
"""


# ============================================================
# PRONUNCIATION-AWARE CARTESIA TTS
# ============================================================


class _PronounceStream:
    """Wrap a Cartesia SynthesizeStream and apply pronunciation overrides."""

    def __init__(self, inner):
        self._inner = inner

    def push_text(self, token: str) -> None:
        self._inner.push_text(_apply_pronunciation(token))

    def end_input(self) -> None:
        self._inner.end_input()

    def flush(self) -> None:
        self._inner.flush()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def __aenter__(self):
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self._inner.__aexit__(*args)

    def __aiter__(self):
        return self._inner.__aiter__()

    async def __anext__(self):
        return await self._inner.__anext__()


class PronounceTTS(cartesia.TTS):
    """Cartesia TTS with pronunciation overrides."""

    def synthesize(self, text: str, *, conn_options=None):
        return super().synthesize(
            _apply_pronunciation(text),
            conn_options=conn_options,
        )

    def stream(self, *, conn_options=None):
        return _PronounceStream(super().stream(conn_options=conn_options))


# ============================================================
# OFFICIAL GMU WEBSITE LOOKUP
#
# Lightweight official-site-only extraction.
# No RAG/vector database required.
# ============================================================

ALLOWED_WEB_HOSTS = {
    "gmu.ac.in",
    "www.gmu.ac.in",
    "ra-gmu.netlify.app",
}

MAX_PAGE_CHARS = 9000
MAX_CANDIDATES = 5


class _SimpleHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.links = []
        self._skip_depth = 0
        self._current_link = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

        if tag == "a" and "href" in attrs:
            self._current_link = attrs["href"]

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

        if tag == "a":
            self._current_link = None

    def handle_data(self, data):
        if self._skip_depth:
            return

        text = re.sub(r"\s+", " ", data).strip()

        if text:
            self.text_parts.append(text)

        if self._current_link:
            self.links.append(self._current_link)

    @property
    def text(self):
        return " ".join(self.text_parts)


def _allowed_url(url: str) -> bool:
    try:
        host = urlparse(url).hostname
        return host in ALLOWED_WEB_HOSTS
    except Exception:
        return False


def _fetch_page(url: str, timeout: float = 4.0):
    if not _allowed_url(url):
        return "", []

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; GMU-Astra/1.0)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )

    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")

        if "text/html" not in content_type:
            return "", []

        raw = response.read(500_000)

    parser = _SimpleHTMLParser()
    parser.feed(raw.decode("utf-8", errors="ignore"))

    return parser.text[:MAX_PAGE_CHARS], parser.links


def _query_terms(query: str):
    stop_words = {
        "what",
        "where",
        "when",
        "which",
        "who",
        "how",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "and",
        "or",
        "can",
        "tell",
        "me",
        "about",
        "please",
        "give",
        "show",
        "does",
        "do",
        "i",
        "we",
        "you",
        "current",
        "latest",
        "today",
    }

    words = re.findall(r"[a-z0-9]+", query.lower())

    return [word for word in words if len(word) >= 3 and word not in stop_words]


def _score_page(url: str, text: str, query: str) -> int:
    terms = _query_terms(query)

    haystack = (url + " " + text[:30000]).lower()

    score = 0

    for term in terms:
        score += haystack.count(term)

        if term in url.lower():
            score += 8

    return score


# ============================================================
# ASTRA AGENT
# ============================================================


class DefaultAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=SYSTEM_PROMPT,
        )

    @function_tool()
    async def greet(
        self,
        context: RunContext,
    ) -> str:
        """
        SPECIAL GREET INTENT.

        Call this function when the user says "greet" or asks Astra
        to greet/welcome everyone for the AI Lab inauguration.

        The function returns the complete ceremonial greeting.
        """

        del context

        logger.info("SPECIAL GREET INTENT TRIGGERED")

        return build_inauguration_greeting()

    @function_tool()
    async def get_gmu_information(
        self,
        context: RunContext,
        query: str,
    ) -> str:
        """
        Search official GM University websites for current, missing,
        detailed, or uncertain GMU information.

        Only official GMU domains are searched.
        """

        del context

        query = query.strip()

        if not query:
            return "No search query was provided."

        logger.info("GMU website lookup: %s", query)

        try:
            homepages = [
                "https://gmu.ac.in/",
                "https://ra-gmu.netlify.app/",
            ]

            discovered = {}

            for homepage in homepages:
                try:
                    text, links = _fetch_page(homepage)

                    if text:
                        discovered[homepage] = text

                    for href in links:
                        absolute = urljoin(homepage, href)

                        if _allowed_url(absolute):
                            discovered.setdefault(absolute, "")

                except Exception as exc:
                    logger.warning(
                        "GMU homepage fetch failed: %s - %s",
                        homepage,
                        exc,
                    )

            candidates = sorted(
                discovered.keys(),
                key=lambda url: _score_page(
                    url,
                    discovered.get(url, ""),
                    query,
                ),
                reverse=True,
            )[:MAX_CANDIDATES]

            results = []

            for url in candidates:
                if not discovered.get(url):
                    try:
                        text, _ = _fetch_page(url)
                        discovered[url] = text

                    except Exception as exc:
                        logger.debug(
                            "Page fetch failed %s: %s",
                            url,
                            exc,
                        )

                text = discovered.get(url, "")

                score = _score_page(
                    url,
                    text,
                    query,
                )

                if text and score > 0:
                    results.append(
                        (
                            score,
                            url,
                            text,
                        )
                    )

            results.sort(
                reverse=True,
                key=lambda item: item[0],
            )

            results = results[:3]

            if not results:
                return (
                    "I could not find a relevant verified answer on the "
                    "official GM University websites. Do not guess the answer."
                )

            parts = ["Official GM University website information found:"]

            for _, url, text in results:
                parts.append(f"SOURCE: {url}\nCONTENT: {text[:5000]}")

            return "\n\n".join(parts)

        except (HTTPError, URLError, TimeoutError) as exc:
            logger.warning(
                "GMU website lookup failed: %s",
                exc,
            )

            return (
                "The official GM University website could not be reached "
                "right now. Do not guess the requested information."
            )

        except Exception as exc:
            logger.exception(
                "Unexpected GMU lookup error: %s",
                exc,
            )

            return (
                "The official GM University lookup failed. "
                "Do not guess the requested information."
            )

    async def on_enter(self):
        await self.session.say(
            "Hello! Welcome to G M U. I'm Astra. How can I help you today?",
            allow_interruptions=True,
        )


# ============================================================
# LIVEKIT SERVER
# ============================================================

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="Astra")
async def entrypoint(ctx: JobContext):

    session = AgentSession(
        stt=groq.STT(
            model="whisper-large-v3-turbo",
            language="en",
        ),
        llm=groq.LLM(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            fallback=nvidia.LLM(
                model="nvidia/nemotron-mini-4b-instruct", temperature=0.4
            ),
        ),
        tts=PronounceTTS(
            model="sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            language="en",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            interruption=InterruptionOptions(
                min_duration=0.8,
                min_words=2,
                resume_false_interruption=True,
                false_interruption_timeout=1.5,
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

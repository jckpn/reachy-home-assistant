from datetime import datetime

from pydantic import BaseModel


class DateTimeInfo(BaseModel):
    year: int
    month: int
    day: int
    hour: int
    minute: int

    @classmethod
    def from_datetime_object(cls, obj: datetime, /) -> "DateTimeInfo":
        return DateTimeInfo(
            year=obj.year,
            month=obj.month,
            day=obj.day,
            hour=obj.hour,
            minute=obj.minute,
        )

    @classmethod
    def now(cls) -> "DateTimeInfo":
        now = datetime.now()
        return cls.from_datetime_object(now)


class AgentNote(BaseModel):
    dt: DateTimeInfo
    note: str


class AgentNotes(BaseModel):
    notes: list[AgentNote]


class AgentNotepad:
    def __init__(self, path: str) -> None:
        self._notes = self._load_notes(path)

    def add_note(self, note: str) -> None:
        agent_note = AgentNote(dt=DateTimeInfo.now(), note=note)
        self._notes.notes.append(agent_note)
        self._write_out()

    def _write_out(self) -> None:
        with open("charlie_memories.txt", "w") as f:
            json = self._notes.model_dump_json(indent=4)
            f.write(json)

    @staticmethod
    def _load_notes(path: str) -> AgentNotes:
        try:
            with open(path, "r") as f:
                json = f.read()
                return AgentNotes.model_validate_json(json)
        except FileNotFoundError:
            return AgentNotes(notes=[])

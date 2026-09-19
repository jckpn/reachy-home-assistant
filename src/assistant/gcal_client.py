import os
from datetime import UTC, datetime
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


class GCalClient:
    def __init__(self) -> None:
        scopes = ["https://www.googleapis.com/auth/calendar.readonly"]
        creds = self._load_creds(scopes)
        self._service = build("calendar", "v3", credentials=creds)

    @staticmethod
    def _load_creds(scopes: list[str]) -> Any | None:
        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", scopes)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", scopes
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        return creds

    def get_upcoming_events(self, max_events: int = 10) -> ...:
        now = datetime.now(tz=UTC).isoformat()
        res = (
            self._service.events()
            .list(
                calendarId="primary",
                timeMin=now,
                maxResults=max_events,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        full_event_details = res.get("items", [])
        basic_event_details = [
            {
                "start": e["start"].get("dateTime") or e.get("date"),
                "end": e["end"].get("dateTime") or "all day event",
                "summary": e["summary"],
            }
            for e in full_event_details
        ]

        return basic_event_details

    @staticmethod
    def _parse_iso(iso: str | None) -> str | None:
        if not iso:
            return None
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    gcal = GCalClient()
    events = gcal.get_upcoming_events()
    print(events)

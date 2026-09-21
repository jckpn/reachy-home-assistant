import os
from datetime import UTC, datetime
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


class GCalClient:
    def __init__(self) -> None:
        creds = self._load_creds(
            scopes=["https://www.googleapis.com/auth/calendar.readonly"],
            token_save_path=os.path.expanduser("~/.reachy-assistant/gcal_token.json"),
            creds_load_path=os.path.expanduser("~/.reachy-assistant/gcal_creds.json"),
        )
        self._service = build("calendar", "v3", credentials=creds)

    @staticmethod
    def _load_creds(
        *,
        scopes: list[str],
        token_save_path: str,
        creds_load_path: str,
    ) -> Any | None:
        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists(token_save_path):
            creds = Credentials.from_authorized_user_file(token_save_path, scopes)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    creds_load_path, scopes
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(token_save_path, "w") as token:
                token.write(creds.to_json())

        return creds

    def get_upcoming_events(
        self,
        *,
        max_events: int = 10,
        keyword_filter: str | None = None,
    ):
        now = datetime.now().isoformat()
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
        basic_event_details = {
            "events": [
                {
                    "start": e["start"].get("dateTime") or e.get("date"),
                    "end": e["end"].get("dateTime") or "all day event",
                    "summary": e["summary"],
                }
                for e in full_event_details
            ]
        }

        if keyword_filter:
            kw = keyword_filter.lower()
            basic_event_details = {
                "events": [
                    e
                    for e in basic_event_details["events"]
                    if kw in e["summary"].lower()
                ]
            }

        return basic_event_details

    @staticmethod
    def _parse_iso(iso: str | None) -> str | None:
        if not iso:
            return None
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

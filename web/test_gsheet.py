import os
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import traceback

# --- Service-Account laden ---
if not os.path.exists("service_account.json"):
    print("service_account.json nicht gefunden!")
    exit()

with open("service_account.json") as f:
    creds_json = f.read()

try:
    creds_dict = json.loads(creds_json)
except Exception as e:
    print("Fehler beim Laden der JSON:", e)
    exit()

client_email = creds_dict.get("client_email")
if not client_email:
    print("'client_email' fehlt in der JSON!")
else:
    print("Service-Account E-Mail:", client_email)

# --- Google Sheets API Scope ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# --- Verbindung autorisieren ---
try:
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    print("Service-Account autorisiert")
except Exception as e:
    print("Fehler bei der Authentifizierung:")
    traceback.print_exc()
    exit()

# --- Sheet-ID eintragen ---
sheet_id = "12cpUeHKpbVEOe_VODjUquNXo3SIbaR3t4mIsnGT_JpQ"

# --- Sheet öffnen ---
try:
    sheet = client.open_by_key(sheet_id).sheet1
    print("Sheet erfolgreich geöffnet:", sheet.title)
except Exception as e:
    print("Fehler beim Öffnen des Sheets:")
    traceback.print_exc()


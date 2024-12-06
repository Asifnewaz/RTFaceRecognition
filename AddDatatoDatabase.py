import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancerealtime-9e847-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "1446896":
        {
            "name": "Asif Newaz",
            "major": "HIS",
            "starting_year": 2024,
            "total_attendance": 0,
            "standing": "3",
            "year": 1,
            "last_attendance_time": "2024-12-05 10:55:00"
        },
    "1420867":
        {
            "name": "Mohsina Binte Asad",
            "major": "HIS",
            "starting_year": 2024,
            "total_attendance": 0,
            "standing": "1",
            "year": 1,
            "last_attendance_time": "2024-12-05 10:55:00"
        },
    "1547470":
        {
            "name": "Saiful Islam",
            "major": "HIS",
            "starting_year": 2024,
            "total_attendance": 0,
            "standing": "2",
            "year": 1,
            "last_attendance_time": "2024-12-05 10:55:00"
        }
}

for key, value in data.items():
    ref.child(key).set(value)
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("secrets/serviceAccountKey2.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancerealtime-9e847-default-rtdb.firebaseio.com/"
})

ref = db.reference('Attendance')

data = {
    "class_id":
        [
            {
                "name": "Asif Newaz",
                "ID" : 1446896,
                "major": "HIS",
                "Date": "01.01.2025",
                "Time": "10:20 AM",
                "Course ID": 0
            },
            {
                "name": "Mohsina Binte Asad",
                "ID" : 1446896,
                "major": "HIS",
                "Date": "01.01.2025",
                "Time": "10:20 AM",
                "Course ID": 0
            },
            {
                "name": "Saiful Islam",
                "ID" : 1446896,
                "major": "HIS",
                "Date": "01.01.2025",
                "Time": "10:20 AM",
                "Course ID": 0
            }
        ]
}

for key, value in data.items():
    ref.child(key).set(value)
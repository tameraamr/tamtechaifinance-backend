import sqlite3

conn = sqlite3.connect('test.db')
cursor = conn.cursor()

# Check if calendar_events table exists
cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name="calendar_events"')
result = cursor.fetchone()
print('calendar_events table exists:', result is not None)

if result:
    # Check how many records
    cursor.execute('SELECT COUNT(*) FROM calendar_events')
    count = cursor.fetchone()[0]
    print('Records in calendar_events:', count)

    # Show first few records
    cursor.execute('SELECT name, date_time, importance FROM calendar_events ORDER BY date_time LIMIT 3')
    records = cursor.fetchall()
    print('First 3 events:')
    for record in records:
        print(f'  {record}')

conn.close()
import psycopg2


def getPostgresConn():
    try:
        conn = psycopg2.connect("dbname='postgis_25_sample' user='postgres' host='localhost' password='forever'")
    except:
        print('I am unable to connect to the database')
    return conn


def executeScript(script):
    conn = getPostgresConn()
    cur = conn.cursor()
    try:
        cur.execute(script)
    except:
        print('I can''t execute the script!')

    rows = cur.fetchall()
    namedict = (
        {
            "first_name": "Joshua",
            "last_name": "Drake"
        },
        {
            "first_name": "Steven",
            "last_name": "Foo"
        },
        {
            "first_name": "David",
            "last_name": "Bar"
        }
    )
    cur.executemany("""INSERT INTO bar(first_name,last_name) VALUES (%(first_name)s, %(last_name)s)""", namedict)
    return

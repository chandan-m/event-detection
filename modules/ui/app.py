from flask import Flask, render_template
import sqlite3

app = Flask(__name__)

DATABASE = "modules/ui/templates/analysis_results.db"

@app.route('/v1')
@app.route('/v1/home')
@app.route('/v1/index')
def homev1():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()
    cur.execute("select * from results order by id")

    rows = cur.fetchall();
    return render_template('homev1.html', rows=rows)


@app.route('/')
@app.route('/v2')
@app.route('/v2/home')
@app.route('/v2/index')
def homev2():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()
    cur.execute("select * from results order by id")

    rows = cur.fetchall();
    return render_template('homev2.html', rows=rows)

@app.route('/tweets/<cluster_id>')
def tweets(cluster_id):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()
    sql = "select distinct tweet from tweets where c_id="+cluster_id+".0"
    cur.execute(sql)

    rows = cur.fetchall();
    return render_template('tweets.html', cluster_id=1, tweets = rows)


@app.route('/test')
def test(sql_file):
    return render_template('test.html')


def run_ui(db_name):
    DATABASE = db_name
    print(DATABASE)
    app.run(debug=True)

if __name__ == '__main__':
    run_ui("../../output/analysis_results.db")    
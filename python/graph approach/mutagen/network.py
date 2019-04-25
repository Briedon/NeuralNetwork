__author__ = "briedmar"

import sys, os

import mysql.connector




if __name__ == '__main__':
    config = {
        'host': 'relational.fit.cvut.cz',
        'port': 3306,
        'user': 'guest',
        'password': 'relational',
        'charset': 'utf8',
        'use_unicode': True,
        'get_warnings': True,
    }

    db = mysql.connector.Connect(**config)
    print(db)
    cur =db.cursor()
    stmt_select = "SELECT * FROM mutagenesis.atom"
    cur.execute(stmt_select)
    print(cur)
    rows = cur.fetchall()
    print(rows)
    db.close()
    print('end\n')


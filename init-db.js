db = db.getSiblingDB("pulsex_db")
db.pulsex_tb.drop()

db.pulsex_tb.insertMany(result)

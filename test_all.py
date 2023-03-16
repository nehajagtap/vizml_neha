from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')
X = [["Date"], ["type"], ["name"], ["gender"], ["crime"], ["status"], ["city"], ["address"], ["duration"], ["postcode"], ["year"], ["weekdays"], ["monthdays"], ["month"], ["number"], ["age"],["cordinates"],["relationship"], ["population"], ["state"], ["school_name"], ["bank_name"], ["company_name"], ["ratio"], ["country"], ["description"], ["id"], ["region"], ["continent"], ["district"], ["float"], ["integer"]]
enc.fit(X)
a= enc.transform([["Date"], ["type"]]).toarray()
print (a)
b= enc.inverse_transform([a[0]])
print(b)
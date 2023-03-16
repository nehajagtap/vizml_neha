# # import nltk
# # from nltk.corpus import wordnet
# # from nltk.stem import WordNetLemmatizer
# # nltk.download('wordnet')
# # def get_synonyms(word):
# #     synonyms = []
# #     for syn in wordnet.synsets(word):
# #         for lemma in syn.lemmas():
# #             synonyms.append(lemma.name())
# #     return synonyms

# # lemmatizer = WordNetLemmatizer()
# # word = lemmatizer.lemmatize("running")
# # synonyms = get_synonyms(word)
# # print(synonyms)



# # # date_str = ["date","birthdate","year","dob","doe","yr"]
# # # nt_date =["company","companies","business","school","schools","population","state", "st","relationship","relations","type","name","gender","sex","crime", "fraud","status","cities","city","addr","address","street","duration","postalcode","postcode","zip","code","day","month","number","num","no","age","lat","lon","latitude","longitude"]

# # # def get_nt_str(nt_list):
# # #     total_list =["company","companies","business","school","schools","population","state", "st","relationship","relations","date","birthdate","dob","doe","type","name","gender","sex","crime", "fraud","status","cities","city","addr","address","street","duration","postalcode","postcode","zip","code","year","yr","day","month","number","num","no","age","lat","lon","latitude","longitude"]
# # #     for i in nt_list:
# # #         total_list.remove(i)
# # #     return total_list

# # # removed_list = get_nt_str(date_str)






from owlready2 import *
onto = get_ontology("http://test.org/onto.owl")

with onto:
    #our entities are classes
    class Coffee(Thing): pass
    
    #related information can also be captured as classes
    class Roast(Thing): pass
    
    #subclassing Roast to break down additional details
    class Dark_Roast(Roast): pass
    class Blonde_Roast(Roast): pass
    class Medium_Roast(Roast): pass
    
    
    class Region(Thing): pass
    
    class Latin_America(Region): pass
    class Asia_Pacific(Region): pass
    class Multi(Region): pass

    #defining the relationship between coffee and roast
    class has_roast(ObjectProperty, FunctionalProperty):
        domain = [Coffee]
        region = [Roast]


    #FunctionalProperties mean it can only be related to one; these coffees can only be grown in one region
    class from_region(Coffee >> Region, FunctionalProperty):
        pass

    #defining the characteristics for a specific coffee type or line
    class Veranda(Coffee):
        equivalent_to = [Coffee & has_roast.value(Blonde_Roast) & from_region.some(Region) & 
                        from_region.only(Latin_America)]

    #.some means it must be related to a Region
    #.only means if it's related to a region it must be the one defined
    class Pike(Coffee):
        equivalent_to = [Coffee & has_roast.value(Medium_Roast) & from_region.some(Region) &
                        from_region.only(Latin_America)]

        
    #defining some unknown coffees and their characteristics
    coffee1 = Coffee(has_roast = Blonde_Roast, from_region=Latin_America())
    coffee2 = Coffee(has_roast = Medium_Roast, from_region=Latin_America())


    sync_reasoner()
    


# print("Classes:")
# for cls in onto.classes():
#     print(" - ", cls)

# print("Properties:")
# for prop in onto.object_properties():
#     print(" - ", prop)

# print("Individuals:")
# for ind in onto.individuals():
#     print(" - ", ind)


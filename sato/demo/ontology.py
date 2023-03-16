from owlready2 import *
import networkx as nx
import matplotlib.pyplot as plt

onto = get_ontology("http://test.org/onto.owl")

with onto:

    class Event(Thing):
        pass

    class Unit(Thing):
        pass

    class Location(Thing):
        pass

    class Crime(Thing):
        pass

    class Person(Thing):
        pass
    
    
    class date_of_crime(DataProperty):
        range = [datetime.datetime]
        domain = [Crime]

    class has_Event(ObjectProperty):
        domain = [Unit]
        range = [Event]

    class has_location(ObjectProperty):
        domain = [Crime]
        range = [Location]

    class has_latitude(DataProperty):
        range = [float]
        domain = [Location]

    class has_longitude(DataProperty):
        range = [float]
        domain = [Location]

    class has_status(DataProperty):
        range = [str]
        domain = [Crime]

    class has_crime(ObjectProperty):
        domain = [Event]
        range = [Crime]

    class has_type(DataProperty):
        range = [str]
        domain = [Person]

    class has_person(ObjectProperty):
        domain = [Crime]
        range = [Person]

    class has_gender(DataProperty):
        range = [str]
        domain = [Person]

    class date_of_birth(DataProperty):
        range = [datetime.datetime]
        domain = [Person]

    class has_age(DataProperty):
        range = [int]
        domain = [Person]

    class has_relationship(ObjectProperty):
        domain = [Person]
        range = [str]


print("Classes:")
for cls in onto.classes():
    print(" - ", cls)

print("Properties:")
for prop in onto.object_properties():
    print(" - ", prop)

print("Individuals:")
for ind in onto.individuals():
    print(" - ", ind)


nodes = ["Event", "Unit", "Location", "Crime", "Person"]
edges = [("Unit", "Event", "has_unit"),
         ("Event", "Crime", "has_crime"),         
         ("Crime", "Location", "has_location"),         
         ("Location", "Latitude", "has_latitude"),         
         ("Location", "Longitude", "has_longitude"),        
         ("Crime", "Status", "has_status"),
         ("Crime", "Person", "has_person"),
         ("Crime", "Date_of_Crime", "Date_of_Crime"),               
         ("Person", "Type", "has_type"),                 
         ("Person", "Gender", "has_gender"),         
         ("Person", "Date of birth", "date_of_birth"),         
         ("Person", "Age", "has_age"),         
         ("Person", "Relationship with victim", "has_relationship")]

G = nx.DiGraph()

for node in nodes:
    G.add_node(node, label=node, shape="rectangle")

for edge in edges:
    G.add_edge(edge[0], edge[1], label=edge[2])

pos = nx.spring_layout(G)

nx.draw(G, pos, with_labels=True)

edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()
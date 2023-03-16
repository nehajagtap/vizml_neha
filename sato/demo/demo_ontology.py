from owlready2 import *
import networkx as nx
import matplotlib.pyplot as plt

onto = get_ontology("http://test.org/onto.owl")

with onto:
    
    #our entities are classes
    class Event(Thing): pass

    #related information can also be captured as classes

    class Crime(Thing): pass
    #subclassing Crime to break down additional details
    class Murder(Crime): pass
    class Theft(Crime): pass
    class Robbery(Crime): pass

    class Location(Thing): pass
    #subclassing Location to break down additional details
    class Germany(Location): pass
    class Pakistan(Location): pass
    class China(Location): pass
    class India(Location): pass
    class Long(): pass
    class Lat(): pass

    class has_crime(ObjectProperty,FunctionalProperty):
        domain = [Event]
        range = [Crime]

    class has_location(Event >> Location, FunctionalProperty): 
        pass

    class shop_robbery(Event): 
        equivalent_to = [Event & has_crime.value(Robbery) & has_location.some(Location) & 
                    has_location.only(Germany)]
        
    class adult_murder(Event): 
        equivalent_to = [Event & has_crime.value(Murder) & has_location.some(Location) & 
                    has_location.only(India)]

    #defining some unknown coffees and their characteristics
    event1 = Event(has_crime = Murder, has_location=India())
    event2 = Event(has_crime = Robbery, has_location=Germany())

sync_reasoner()
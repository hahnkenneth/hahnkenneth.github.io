---
layout: page
title: Sustainable Food Delivery Service
description: Neo4j graph databases
img: assets/img/ebike_background.jpg
importance: 1
category: Data Engineering
related_publications: false
---

Throughout the course of my Fundamentals of Data Engineering class, we have been developing creating a delivery service for our hypothetical company called AGM (Acme Gourmet Meals). Prior to starting this project, we created PostgreSQL relational tables that stored `customer` data and `stores` data. For this final project, we were to create a NoSQL Neo4j database for an open-ended project with any of the data from the relational tables.

For my project, I used Neo4j to build a graph database to begin the development of a delivery service in the Bay Area. The graph database would consist of `customer` nodes, `store` nodes, and `station` nodes, where the `station` nodes were representative of the different stations for BART. To add extra challenge to this project, I decided to create a delivery survice that was eco-friendly and so all transportation services would either be through BART or by e-bike to go from a store to a customer.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/bart_map.png" title="graph layout" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Map of Bay Area Rapid Transit (BART) stations
</div>

Below is the high level layout of our graph database, where `store_id`, store `[lat,long]`, `customer_id`, and customer `[lat,long]` were given through relational databases, where `[lat,long]` is the latitude and longitude coordinates. The `station` nodes were compiled into a csv and stored in a relational database. Each station was comprised of several nodes, representing the different colored lines on BART. They stations were related by `transfer` and `commute` times, where the tranfer time is the average time to transfer from one line to another at the same station, and commute time is the average time to go from one station to the next station.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/agm_delivery_graph.jpg" title="graph layout" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    High level overview of the graph database connecting the stores, BART stations, and customers.
</div>

The two main pieces of information that we did not have were the time to e-bike from one node to another and also the customer `[lat,long]` coordinates. The customer `[lat,long]` coordinates were based off a random coordinate within a geodesic box, using the `geographiclib.geodesic` library, shown in the code below:

{% raw %}

```python
    def my_calculate_box(point, miles):
        "Given a point and miles, calculate the box in form left, right, top, bottom"
        
        geod = Geodesic.WGS84

        kilometers = miles * 1.60934
        meters = kilometers * 1000

        g = geod.Direct(point[0], point[1], 270, meters)
        left = (g['lat2'], g['lon2'])

        g = geod.Direct(point[0], point[1], 90, meters)
        right = (g['lat2'], g['lon2'])

        g = geod.Direct(point[0], point[1], 0, meters)
        top = (g['lat2'], g['lon2'])

        g = geod.Direct(point[0], point[1], 180, meters)
        bottom = (g['lat2'], g['lon2'])
        
        return(left, right, top, bottom)
    
    # Find a geographic box for each customer based on the zip code that they live in.
    # zip_center_point is the [lat,long] of the central point in the zip code
    # zip_box_miles is length of the zip code in miles, based on the area of the zip code
    # multiply by np.sqrt(2)/2 to get half diagonal of the square to create the box
    customers_df['customer_box'] = customers_df.apply(lambda row: my_calculate_box(row['zip_center_point']
                                                                               ,row['zip_box_miles']*np.sqrt(2)*0.5),axis=1)

    # create a random latitude and longitude for each customer based on the box created
    customers_df['lat_long'] = customers_df.apply(lambda row: (random.uniform(row['customer_box'][3][0]
                                                                                ,row['customer_box'][2][0])
                                                                 ,random.uniform(row['customer_box'][0][1]
                                                                               ,row['customer_box'][1][1]))
                                                     ,axis = 1) 
    
```

{% endraw %}

The e-bike times were created with the Google Maps Directions API using the `[lat,long]` coordinates of all the nodes. Then we applied this to all the customers and stores to connect them to Bart Stations and to one another. Google Maps Directions gave us the durations between two nodes via biking; however, since we were using e-bikes, we applied a coorection factor of 1.5, assuming that e-bikes were 50% faster than normal bikes. This assumption is reasonable since e-bikes can have maximum speeds of up to 28 mph, which is double the speed of a normal bike. The code for the functions and application are shown below:

{% raw %}

```python
    def google_api_bike_travel_time(api_key, origin, destination):
        """Use Google Maps API for a given key to get the biking travel time between the origin [lat,long]
        and the destination [lat,long]. Divides  that travel time by the e_bike_factor, correcting for the speed
        of an e-bike over normal biking."""
        base_url = "https://maps.googleapis.com/maps/api/directions/json?"
        params = {
            'origin': f"{origin[0]},{origin[1]}",  # Latitude, Longitude
            'destination': f"{destination[0]},{destination[1]}",  # Latitude, Longitude
            'mode': 'bicycling',
            'key': api_key
        }
        
        response = requests.get(base_url, params=params)
        directions = response.json()
        
        # assume e-bikes are 50% faster than normal bikes
        e_bike_factor = 1.5
        
        if directions['status'] == 'OK':
            route = directions['routes'][0]
            leg = route['legs'][0]
            duration = leg['duration']['value'] 
            return duration / e_bike_factor
        else:
            return "Invalid Location"
    

    def find_biking_time(thing,store_or_bart="bart",manual=0):
        """Used to find conduct the API call to find the biking times between a thing and either to the bart or to
        the store. Only used for the .apply() function on a pandas dataframe."""
        if store_or_bart == "bart":
            origin = thing['lat_long']
            destination = thing['nearest_bart_lat_long']
        elif store_or_bart == "store":
            origin = thing['lat_long']
            destination = thing['nearest_store_lat_long']
        return google_api_bike_travel_time(my_api_key,origin,destination)
    
    customers_df['nearest_store_bike_time_seconds'] = customers_df.apply(lambda row: find_biking_time(row,"store"),axis=1)
    stores_df['nearest_bart_bike_time_seconds'] = stores_df.apply(lambda row: find_biking_time(row,"bart"),axis=1)

```

{% endraw %}

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>



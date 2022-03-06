# Step 1: extract hand coordinates (try only palm and palm with thumb/index tip)
* Title - ```id_class.csv```
* Hand - ```(1 or 2)```
* Coordinates x and y OR x/y change

# Step 2: Split into 15-20 equally spread frames
only keep enough data to keep the hand movement but keep the shape small

# Step 3: Clean data
ensure there's always 2 hands by duplicating hand 2 coordinates if null, showing no change

# Step 4: Define model shape
## Uses Sequential, therefore only one input and one output per layer
* for palm only - ```1 layer per hand, palm coordinate change```
* for 3 points - ```1 layer per point, 6 total, coordinate change```

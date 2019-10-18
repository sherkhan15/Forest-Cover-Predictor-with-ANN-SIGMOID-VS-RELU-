# Forest-Cover-Normalized-UnNormalized-Predictor-with-ANN
Context This dataset contains tree observations from four areas of the Roosevelt National Forest in Colorado.
All observations are cartographic variables (no remote sensing) from 30 meter x 30 meter sections of forest. 
There are over half a million measurements total!
Content This dataset includes information on tree type, shadow coverage, distance to nearby landmarks (roads etcetera), soil type, and local topography.
The original database owners are Jock A. Blackard, Dr. Denis J. Dean, and Dr. Charles W. Anderson of the Remote Sensing and GIS Program at Colorado State University.
Inspiration Can you build a model that predicts 
what types of trees grow in an area based on the surrounding characteristics?
 
  * What kinds of trees are most common in the Roosevelt National Forest?
  * Which tree types can grow in more diverse environments? 
  * Are there certain tree types that are sensitive to an environmental factor, such as elevation or soil type?

The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch. You are asked to predict an integer classification for the forest cover type. The seven types are:

1 - Spruce/Fir

2 - Lodgepole Pine

3 - Ponderosa Pine

4 - Cottonwood/Willow

5 - Aspen

6 - Douglas-fir

7 - Krummholz


The training set (15120 observations) contains both features and the Cover_Type. The test set contains only the features. You must predict the Cover_Type for every row in the test set (565892 observations).


* Data Fields
* Elevation - Elevation in meters
* Aspect - Aspect in degrees azimuth
* Slope - Slope in degrees
* Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
* Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
* Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
* Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
* Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
* Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
* Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points
* Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
* Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
* Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation

**The wilderness areas are:**

*  Rawah Wilderness Area
*  Neota Wilderness Area
*  Comanche Peak Wilderness Area
*  Cache la Poudre Wilderness Area

**The soil types are:

*   Cathedral family - Rock outcrop complex, extremely stony.

*   Vanet - Ratake families complex, very stony.

*   Haploborolis - Rock outcrop complex, rubbly.

*   Ratake family - Rock outcrop complex, rubbly.

*   Vanet family - Rock outcrop complex complex, rubbly.

*   Vanet - Wetmore families - Rock outcrop complex, stony.

*   Gothic family.

*   Supervisor - Limber families complex.

*   Troutville family, very stony.

*   Bullwark - Catamount families - Rock outcrop complex, rubbly.

*   Bullwark - Catamount families - Rock land complex, rubbly.

*   Legault family - Rock land complex, stony.

*   Catamount family - Rock land - Bullwark family complex, rubbly.

*   Pachic Argiborolis - Aquolis complex.

*   unspecified in the USFS Soil and ELU Survey.

*   Cryaquolis - Cryoborolis complex.

*   Gateview family - Cryaquolis complex.

*   Rogert family, very stony.

*   Typic Cryaquolis - Borohemists complex.

*   Typic Cryaquepts - Typic Cryaquolls complex.

*   Typic Cryaquolls - Leighcan family, till substratum complex.

*   Leighcan family, till substratum - Typic Cryaquolls complex.

*   Leighcan family, till substratum, extremely bouldery.

*   Leighcan family, extremely stony.

*   Leighcan family, warm, extremely stony.

*   Granile - Catamount families complex, very stony.

*   Leighcan family, warm - Rock outcrop complex, extremely stony.

*   Leighcan family - Rock outcrop complex, extremely stony.

*   Como - Legault families complex, extremely stony.

*   Como family - Rock land - Legault family complex, extremely stony.

*   Leighcan - Catamount families complex, extremely stony.

*   Catamount family - Rock outcrop - Leighcan family complex, extremely stony.

*   Leighcan - Catamount families - Rock outcrop complex, extremely stony.

*   Cryorthents - Rock land complex, extremely stony.

*   Cryumbrepts - Rock outcrop - Cryaquepts complex.

*   Bross family - Rock land - Cryumbrepts complex, extremely stony.

*   Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.

*   Leighcan - Moran families - Cryaquolls complex, extremely stony.

*   Moran family - Cryorthents - Leighcan family complex, extremely stony.

*   Moran family - Cryorthents - Rock land complex, extremely stony.


sensor
Pre-extracted CANBUS data:

0. accel_pedal_info.csv
1. rtk_pos_info.csv
2. steer_info.csv
3. vel_info.csv
4. brake_pedal_info.csv
5. rtk_track_info.csv
6. turn_signal_info.csv
7. yaw_info.csv

########################################################

target
Goal-oriented action involves the driver’s manipulation of the vehicle in a navigation task:

0. Background
1. intersection passing
2. left turn
3. right turn
4. left lane change
5. right lane change
6. left lane branch
7. right lane branch
8. crosswalk passing
9. railroad passing
10. merge
11. U-turn


########################################################


L0 - Operation_Goal-oriented

0.  [0]  right turn
1.  [1]  intersection passing
2.  [2]  merge
3.  [3]  left lane change
4.  [4]  right lane branch
5.  [5]  right lane change
6.  [6]  intersection passing
7.  [7]  left turn
8.  [8]  crosswalk passing
9.  [9]  park
10. [10] railroad passing
11. [11] left lane branch
12. [12] U-turn
13. [13] park park
14. [14] 
15. [15] Park park

########################################################

L1 - Cause

16. [0]  congestion
17. [1]  Sign
18. [2]  red light
19. [3]  crossing vehicle
20. [4]  Parked vehicle
21. [5]  yellow light
22. [6]  crossing pedestrian
23. [7]  merging vehicle
24. [8]  on-road bicyclist
25. [9]  pedestrian near ego lane
26. [10] park
27. [11] on-road motorcyclist
28. [12] vehicle cut-in
29. [13] road work
30. [14] turning vehicle
31. [15] vehicle passing with lane departure

########################################################

L2 - Area

32. [1]  downtown
33. [2]  freeway
34. [3]  tunnel

########################################################

L3 - Attention

35. red light
36. on-road motorcyclist
37. vehicle cut-in
38. crossing vehicle
39. Sign
40. merging vehicle
41. congestion
42. Parked vehicle
43. yellow light
44. crossing pedestrian
45. road work
46. on-road bicyclist
47. pedestrian near ego lane
48. vehicle passing with lane departure
49. start
50. end
51. roundabout
52. 
53. highway exit
54. While turning left, the frontal vehicle is also turning left slowly as there is a crossing pedestrian
55. While U-turning, the ego car is waiting for oncoming vehicles
56. atypical
57. driveway
58. Long Merge
59. Atypical
60. Roundabout
61. Ambiguous as it can interpret as left lane branch
62. ramp
63. vehicle on the hilly road
64. atypical T-intersection
65. wheelchair
66. hard example
67. curve road
68. curved road
69. very long right turn
70. or right lane change
71. speical situation
72. hump
73. crossing vehicle
74. on-road bicyclist
75. crossing pedestrian
76. merging vehicle
77. Parked vehicle
78. red light
79. vehicle cut-in
80. yellow light
81. on-road motorcyclist
82. road work
83. Sign
84. pedestrian near ego lane

########################################################

L4 - note

49. [1]  start
50. [2]  end
51. [3]  roundabout
52. [4]  
53. [5]  highway exit
54. [6]  While turning left, the frontal vehicle is also turning left slowly as there is a crossing pedestrian
55. [7]  While U-turning, the ego car is waiting for oncoming vehicles
56. [8]  atypical
57. [9]  driveway
58. [10] Long Merge
59. [11] Atypical
60. [12] Roundabout
61. [13] Ambiguous as it can interpret as left lane branch
62. [14] ramp
63. [15] vehicle on the hilly road
64. [16] atypical T-intersection
65. [17] wheelchair
66. [18] hard example
67. [19] curve road
68. [20] curved road
69. [21] very long right turn
70. [22] or right lane change
71. [23] speical situation
72. [24] hump

########################################################

L5 - Attention 2

73. [1]  crossing vehicle
74. [2]  on-road bicyclist
75. [3]  crossing pedestrian
76. [4]  merging vehicle
77. [5]  Parked vehicle
78. [6]  red light
79. [7]  vehicle cut-in
80. [8]  yellow light
81. [9]  on-road motorcyclist
82. [10] road work
83. [11] Sign
84. [12] pedestrian near ego lane

########################################################

L6 - Operation_Stimuli-driven

85. [1]  stop 4 congestion
86. [2]  stop 4 sign
87. [3]  stop 4 light
88. [4]  Avoid parked car
89. [5]  stop 4 pedestrian
90. [6]  Stop for others
91. [7]  Avoid pedestrian near ego lane
92. [8]  Avoid on-road bicyclist
93. [9]  Avoid TP
94. [10] empty
# Optimizing-Ride-Requests-for-Drivers-using-Reinforcement-Learning
The project focuses on developing reinforcement learning (RL) algorithms to assist cab drivers in making optimal decisions regarding ride acceptance to maximize their fare earnings.


By utilizing the TLC NYC yellow cab rides dataset and constructing a custom RL environment, we are simulating the dynamic nature of demand and resource availability within the context of cab driving. Through the extraction of relevant features and the definition of an action space, we are laying the groundwork for instantaneous decision-making regarding ride acceptance.
We aim to construct a robust RL environment tailored to the unique challenges faced by cab drivers in urban settings. Cab drivers operating in urban environments encounter significant hurdles in maximizing fare earnings due to suboptimal decision-making processes. The
intricacies of balancing ride acceptance, pickup locations, and time management present a multifaceted problem.
Dataset
The project leveraged the TLC NYC yellow cab rides dataset, comprising real-world trip data collected throughout January. Key features such as pickup time, dropoff time, day of the week, pickup zone, dropoff zone, and total fare were extracted to construct a realistic simulation environment. By utilizing this dataset, the project effectively modeled ride requests and driver
actions within the environment, enhancing the relevance and applicability of its findings. Additionally, rewards from the dataset were normalized to calculate the effective penalty, ensuring accurate evaluation of algorithm performance.

Environment Description:
Defined environment with state space comprising pickup zone, time of day (morning, afternoon, evening, night), and day of the week.
State Space:

a) Zone – This is the driver’s current zone.

b) Time of the Day – This is the time zone of the day where the day is divided into morning, afternoon, evening, night.

c) Day of the week – This is the day of the week starting from Monday till Sunday. 


Actions Space:

a) Action 0, Zone - Wait: This action represents the driver wanting to wait in the same zone for 15 minutes.

b) Action 1, Zone - Accept Trip: This action represents the driver accepting the next quickest ride within the same zone, if the drop location is in a different zone, then we reduce the fare by a negative penalty.

c) Action 2 , Zone – Move to another zone: This action represents the driver wanting to move to another zone without taking a ride.

Total of 267 zones representing areas across Bronx, Brooklyn, Manhattan, Queens, and Staten Island accurately reflect geographical diversity of taxi service area.


Rewards:
a) Reward $0 -This is the reward given for waiting in the same zone.

b) Reward $Fare - This is the reward given for completing the ride in the same zone.

c) Reward $Fare – Penalty - This is the reward given for completing the ride in a different zone.

d) Reward $Penalty - This is a negative reward where the driver moves to another zone without a trip.


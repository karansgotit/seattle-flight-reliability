# Feature Availability Audit

**Question:** Is this value known at booking time, before the flight operates?

## Target
| Column | Notes |
|--------|-------|
| ArrDelay | cancelled and diverted flights have NULL ArrDelay because they never arrived at the airport so these rows will be dropped.Therefore this changes the scope of the model because the model won't be trained on said data which means that the model that will be built from this won't account for cancellation or diversion so the model is only relevant if the assumption is "Given that this flight flies and arrives at said airport,what will be the predicted delay?" |

## Pre-Booking
| Column | Notes |
|--------|-------|
| FlightDate | |
| Year | I kept this because then i wont have to generate this column from flight date,it is purely for convenience|
| Quarter |same as year |
| Month |same as year|
| DayofMonth | same as year |
| DayOfWeek | same as year |
| IATA_CODE_Reporting_Airline | |
| Flight_Number_Reporting_Airline | |
| Dest | |
| CRSDepTime | |
| CRSArrTime | |
| CRSElapsedTime | |
| Distance | |

## Post-Flight
| Column | Notes |
|--------|-------|
| DepTime | |
| DepDelay | |
| DepDelayMinutes | |
| DepDel15 | |
| DepartureDelayGroups | |
| TaxiOut | |
| WheelsOff | |
| WheelsOn | |
| TaxiIn | |
| ArrTime | |
| ArrDelayMinutes | |
| ArrDel15 | |
| ArrivalDelayGroups | |
| ActualElapsedTime | |
| AirTime | |
| CarrierDelay | |
| WeatherDelay | |
| NASDelay | |
| SecurityDelay | |
| LateAircraftDelay | |
| FirstDepTime | |
| TotalAddGTime | |
| LongestAddGTime | |
| Cancelled |Cancellation Status is only known after the flight was already due to operate and this also affects which rows have a valid target(Cancelled flights have NULL for ArrDelay) |
| CancellationCode | Only populated when Cancelled=1 and cancelled is known after the flight was due to operate therefore cancellation code is also post-flight|
| Diverted | If a flight is diverted,then it means the target will probably have unusual values since it never arrived at the arrival airport |
| DivAirportLandings | |
| DivReachedDest | |
| DivActualElapsedTime | |
| DivArrDelay | |
| DivDistance | |
| Div1Airport | |
| Div1AirportID | |
| Div1AirportSeqID | |
| Div1WheelsOn | |
| Div1TotalGTime | |
| Div1LongestGTime | |
| Div1WheelsOff | |
| Div1TailNum | |
| Div2Airport | |
| Div2AirportID | |
| Div2AirportSeqID | |
| Div2WheelsOn | |
| Div2TotalGTime | |
| Div2LongestGTime | |
| Div2WheelsOff | |
| Div2TailNum | |
| Div3Airport | |
| Div3AirportID | |
| Div3AirportSeqID | |
| Div3WheelsOn | |
| Div3TotalGTime | |
| Div3LongestGTime | |
| Div3WheelsOff | |
| Div3TailNum | |
| Div4Airport | |
| Div4AirportID | |
| Div4AirportSeqID | |
| Div4WheelsOn | |
| Div4TotalGTime | |
| Div4LongestGTime | |
| Div4WheelsOff | |
| Div4TailNum | |
| Div5Airport | |
| Div5AirportID | |
| Div5AirportSeqID | |
| Div5WheelsOn | |
| Div5TotalGTime | |
| Div5LongestGTime | |
| Div5WheelsOff | |
| Div5TailNum | |

## Drop
| Column | Notes |
|--------|-------|
| Reporting_Airline | |
| DOT_ID_Reporting_Airline | |
| Tail_Number | |
| OriginAirportID | |
| OriginAirportSeqID | |
| OriginCityMarketID | |
| OriginCityName | |
| OriginState | |
| OriginStateFips | |
| OriginStateName | |
| OriginWac | |
| DestAirportID | |
| DestAirportSeqID | |
| DestCityMarketID | |
| DestCityName | |
| DestState | |
| DestStateFips | |
| DestStateName | |
| DestWac | |
| DepTimeBlk | This is just a range on the scheduled departure time that is why it is in the drop column|
| ArrTimeBlk |This is just a range on the Scheduled Arrival time that is why it is in the drop column|
| Flights | |
| DistanceGroup | |
| Origin | All rows are SEA-origin after filtering, so this column is constant and has no predictive value |
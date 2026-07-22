# Data Quality Audit

- **Shape without filtering to SEA-origin flights:** (14080680, 110)
- **Shape after filtering to SEA-origin flights:** (328559, 110)
- **Null counts based on features (Target and Pre-booking):** ArrDelay: 4069
    This is because of cancelled and diverted flights. These values are NULL for such flights because they never arrive at the DEST airport. For this project's scope, I intend to drop these values so the model's scope now becomes to predict the expected delay given the flight operates and arrives at the destination airport.

- **Duplicate counts on what makes a flight unique:** The criteria I used to define what makes a flight unique is having the same FlightDate, IATA_CODE_Reporting_Airline, Flight_Number_Reporting_Airline, Origin, Dest, and according to this criteria there are zero duplicates after filtering for SEA-origin flights.
- **Month Coverage:** I verified using value_counts() on the Year and Month columns of the filtered dataset with SEA-origin flights only whether there are entries for all 24 months across 2 years, and they are all present.
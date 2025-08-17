import pandas as pd
from skyfield.api import load, utc
from datetime import datetime
from framework_pkg.survival_probablity import ParseDate
import numpy as np




# def calculate_day_length(days_since_start, lat=36):
#     """
#     Calculate daylight duration (hours) for a given day and latitude.
    
#     Parameters:
#     -----------
#     days_since_start : float or array-like
#         Days since reference date (2008-09-15).
#     lat : float, optional
#         Latitude in degrees (-90 to 90). Default: 36°.
    
#     Returns:
#     --------
#     daylight_hours : float or array-like
#         Hours of daylight (0 to 24).
    
#     Formula:
#     --------
#     Based on solar declination (δ) and latitude (φ):
#     - δ ≈ arcsin(-sin(ε) * cos(2π(t - t_s)/365.25)), where ε = Earth's obliquity (23.44°).
#     - Daylight = (12/π) * arccos(-tan(φ) * tan(δ)).
#     """
#     # Constants
#     OBLIQUITY_RADIANS = np.radians(23.44)  # Earth's axial tilt (ε)
#     DAYS_PER_YEAR = 365.25
    
#     # Reference dates
#     start_date = pd.Timestamp('2008-09-15')
#     winter_solstice = pd.Timestamp('2008-12-21')
#     t_s = (winter_solstice - start_date).days  # Days to solstice
    
#     # Solar declination (δ)
#     sin_delta = -np.sin(OBLIQUITY_RADIANS) * np.cos(2 * np.pi * (days_since_start - t_s) / DAYS_PER_YEAR)
#     cos_delta = np.sqrt(1 - sin_delta**2)
    
#     # Avoid numerical errors in arccos
#     tan_lat = np.tan(np.radians(lat))
#     tan_delta = sin_delta / cos_delta
#     term = tan_lat * tan_delta
#     term = np.clip(term, -1, 1)  # Ensure arccos domain [-1, 1]
    
#     # Daylight hours (handle polar day/night)
#     daylight_hours = np.where(term >= 1, 0, np.where(term <= -1, 24, (12/np.pi) * np.arccos(term)))
    
#     return daylight_hours



def calculate_day_length(days_since_start, lat=36):
    """
    Calculate daylight duration (hours) for a given day and latitude.
    
    Parameters:
    -----------
    days_since_start : float or array-like
        Days since reference date (2008-09-15).
    lat : float, optional
        Latitude in degrees (-90 to 90). Default: 36°.
    
    Returns:
    --------
    daylight_hours : float or array-like
        Hours of daylight (0 to 24).
    
    Formula:
    --------
    Based on solar declination (δ) and latitude (φ):
    - δ ≈ arcsin(-sin(ε) * cos(2π(t - t_s)/365.25)), where ε = Earth's obliquity (23.44°).
    - Daylight = (12/π) * arccos(-tan(φ) * tan(δ)).
    """
    # Constants
    OBLIQUITY_RADIANS = np.radians(23.44)  # Earth's axial tilt (ε)
    DAYS_PER_YEAR = 365.25
    
    # Reference dates
    start_date = pd.Timestamp('2008-09-15')
    winter_solstice = pd.Timestamp('2008-12-21')
    t_s = (winter_solstice - start_date).days  # Days to solstice
    
    # Solar declination (δ)
    sin_delta = -np.sin(OBLIQUITY_RADIANS) * np.cos(2 * np.pi * (days_since_start - t_s) / DAYS_PER_YEAR)
    cos_delta = np.sqrt(1 - sin_delta**2)
    
    cos_lam = np.cos(np.radians(lat))
    sin_lam = np.sin(np.radians(lat))

    t_day = np.linspace(0,0.5,2000)
    cos_eta = cos_lam * cos_delta * np.cos(2 * np.pi * t_day) - sin_lam * sin_delta

    if len(t_day[cos_eta >= 0]) == len(t_day):
        return 24
    elif  0 < len(t_day[cos_eta >= 0]) < len(t_day)  : 
        return t_day[cos_eta >= 0][-1] * 24
    else:
        return 0

def main():
    time_scale = load.timescale()  # Create a timescale object
    t0 = time_scale.utc(datetime(1970, 1, 1, 0, 0, 0, tzinfo=utc))
    
    first_day = '2008,9,15'
    firstday = ParseDate(first_day)
    zeroday = firstday.tt - t0.tt
    modulation_data = np.loadtxt('./Data/sksolartimevariation5804d.txt')
    modulation_data[:, :3] /= (60. * 60. * 24.)  # Convert time columns to days
    modulation_data = modulation_data[modulation_data[:, 0] - modulation_data[:, 1] >= zeroday]
    modulation_data[:, 0] -= zeroday

    organized_starting_bin = np.column_stack((
        np.floor(modulation_data[:, 0] - modulation_data[:, 1]),
        ((modulation_data[:, 0] - modulation_data[:, 1]) % 1) * 24
    ))

    organized_ending_bin = np.column_stack((
        np.floor(modulation_data[:, 0] + modulation_data[:, 1]),
        ((modulation_data[:, 0] + modulation_data[:, 2]) % 1) * 24
    ))

    number_of_bins = len(organized_starting_bin[:,0])
    organized_data = [[] for i in range (4)]

    for bins in range (number_of_bins):
        
        T_day = 0
        T_night = 0
        t_k = 0
        
        number_of_days = np.arange(int(organized_starting_bin[bins,0]), int(organized_ending_bin[bins,0]) + 1)
    
        if organized_starting_bin[bins,1] > calculate_day_length(number_of_days[0]) :

             if organized_starting_bin[bins,1] > 24 - calculate_day_length(number_of_days[0]):
                 T_day = 0
                 T_night = 24 - organized_starting_bin[bins,1]
             else:
                 T_day = 24 - calculate_day_length(number_of_days[0]) - organized_starting_bin[bins,1]
                 T_night = calculate_day_length(number_of_days[0])   
        else:
             T_day = 24 - 2 * calculate_day_length(number_of_days[0])
             T_night = 2 * calculate_day_length(number_of_days[0]) - organized_starting_bin[bins,1]
        

        
        if T_day >= 0.1  or T_night >= 0.1:    
            t_k = number_of_days[0] #+ (organized_starting_bin[bins,1] + (T_day + T_night) / 2) / 24.
            organized_data[0].append(bins)
            organized_data[1].append(t_k)
            organized_data[2].append(T_day)
            organized_data[3].append(T_night)
        
        for j in (number_of_days[1:-1]):
            organized_data[0].append(bins)
            #organized_data[1].append(j + 0.5)
            organized_data[1].append(j)
            organized_data[2].append(24 - 2 * calculate_day_length(j))
            organized_data[3].append(2 * calculate_day_length(j))

        if organized_ending_bin[bins,1] > calculate_day_length(number_of_days[-1]) :
            if organized_ending_bin[bins,1] > 24 - calculate_day_length(number_of_days[-1]):
                T_day = 24 - 2 * calculate_day_length(number_of_days[-1])
                T_night = 2 * calculate_day_length(number_of_days[-1]) + organized_ending_bin[bins,1] - 24 
            else:
                T_day = organized_ending_bin[bins,1] - calculate_day_length(number_of_days[-1])
                T_night = calculate_day_length(number_of_days[-1])    
        else:
            T_day = 0
            T_night = organized_ending_bin[bins,1]
        
        if T_day >= 0.1  or T_night >= 0.1:     
            t_k = number_of_days[-1] #+ (0.5 * organized_ending_bin[bins,1]) / 24.
            organized_data[0].append(bins)
            organized_data[1].append(t_k)
            organized_data[2].append(T_day)
            organized_data[3].append(T_night)

    organized_data = np.array(organized_data).T

    # Save to a text file
    np.savetxt('./Data/time_exposures_1.txt', organized_data, fmt='   '.join(['%i'] + ['%.1f']*3), header='num_bin  t_k  T_k^day  T_k^night')
    #np.savetxt('./Data/modulation_data.txt', modulation_data[:,[0,3,4,5]], fmt='   '.join(['%.2f']*4))


if __name__ == "__main__":
    main()

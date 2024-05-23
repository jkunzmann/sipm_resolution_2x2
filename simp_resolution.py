# ///////////////////////////////////////////////////////////////// //
#                                                                   //
# Last modifications: 10.05.2024 by Jan Kunzmann, University of Bern//
#                                                                   //
# ///////////////////////////////////////////////////////////////// //

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import os
import json

def open_hdf5_dataset(file_path, dataset_path):
    """
    Opens a specific dataset in an HDF5 file and returns its content in array.

    Args:
    - file_path (str): Path to the HDF5 file.
    - dataset_path (str): Path to the dataset within the HDF5 file.

    Returns:
    - data: Content of the dataset.
    """
    # Open the HDF5 file in read mode
    with h5py.File(file_path, 'r') as file:
        # Open the dataset
        dataset = file[dataset_path]
        # Read the content of the dataset
        data = dataset[:]
        return data

def print_hdf5_structure(file_path):
    """
    Prints the structure of an HDF5 file, including all groups and datasets.

    Args:
    - file_path (str): Path to the HDF5 file.
    """
    # Open the HDF5 file in read mode
    with h5py.File(file_path, 'r') as file:
        # Function to recursively print groups and datasets
        def print_group(name, obj):
            if isinstance(obj, h5py.Group):
                print("Group:", name)
                for key in obj.keys():
                    print_group(name + '/' + key, obj[key])
            elif isinstance(obj, h5py.Dataset):
                print("Dataset:", name)

        # Start printing from the root group
        file.visititems(print_group)

def plot_the_wvf(data, meas_nr, adc, ch):
    """
    Plots the waveform of a single event for a specific adc and channel

        Args:
    - data (array): data array.
    - meas_nr (int): number of measurement to plot
    - adc (int): adc value int
    - ch (int): channel value int
    """
    plt.plot(data[meas_nr,adc,ch,:])
    plt.xlabel('time')
    plt.ylabel('adc count')
    plt.title(f'Waveform of a single event, ch {ch}, measurement nr {meas_nr}, adc {adc}')    
    plt.show()

def integrate_for_gate(data,g_min, g_max):
    """
    Integrates the wave for a specific gate size.

        Args:
    - data (array): data array.
    - g_min (int): gate starting value 
    - g_max (int): gate ending value

        Returns:
    - integrated_data (float) the integrated waveform value between the gate
    """
    integrated_data = np.sum(data[:,:,:,g_min:g_max+1], axis=3)
    return integrated_data

def mean_for_integral(data):
    """
    Calculates the mean value of the integrals for all measurements.

        Args:
    - data (array): data array.

        Returns:
    - mean_data (float) returns the mean of the data
    """
    mean_data = np.mean(data[:,:,:], axis=0)
    return mean_data

def plot_the_int_distr_for_one(data, adc, ch):
    """
    Plots the distribution of the integral values for the data set for a specific adc and channel.

        Args:
    - data (array): data array.
    - adc (int): adc value int
    - ch (int): channel value int
    """
    plt.hist(data[:,adc,ch], bins = 200, density = True)
    plt.xlabel('mean amount of integral per measurement')
    plt.ylabel('#')
    plt.title(f'Integral Distribution of a single event, ch {ch}, adc {adc}')    
    plt.show()

def plot_the_int_distr_for_one_w_fit(data, data_file_name, pdf_file_path, adc, ch, bin_n=200):
    """
    Plots the distribution of the integral values for the data set for a specific adc and channel and fits an gaussian to the data.

        Args:
    - data (array): data array.
    - data_file_name (str): name of the calibration file
    - pdf_file_path (str): Path for the pdf file.
    - adc (int): adc value int
    - ch (int): channel value int
    - bin_n (int): number of pins for the histogram

        Returns:
    - params: [mu, sigma, A] of the fit 
    - covariance: Covariance of the parameters
    """
    hist, bins, _ = plt.hist(data[:,adc,ch], bins = bin_n, density=True)
    #get bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2    
    
    #start parameters
    p0 = estimate_start_parameters(data)
    
    #create the gauss
    def gauss(x, mu, sigma, A):
        return A * np.exp(-0.5 * ((x - mu) / sigma)**2)
    try:
        params, covariance = curve_fit(gauss, bin_centers, hist, p0=p0)
    
        plt.plot(bin_centers, gauss(bin_centers, *params), color='red', label='Gauss-Fit')
        plt.xlabel('mean amount of integral per measurement')
        plt.ylabel('#')
        plt.title(f'Integral Distribution of a single event, ch {ch}, adc {adc}')

        # Generate PDF file name
        pdf_file_name = f'{pdf_file_path}/{data_file_name}_ADC{adc}_CH{ch}_plot_w_fit.pdf'
    
        # Save plot as PDF
        plt.savefig(pdf_file_name)
        print(f"Plot saved as '{pdf_file_name}'.")

        plt.close()
        return params, covariance

    except (RuntimeError, OptimizeWarning):
        print("Fitting failed.")
        p = [-1,-1,-1]
        c =[[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]

        plt.hist(data[:,adc,ch], bins = 300, density=True)

        plt.xlabel('mean amount of integral per measurement')
        plt.ylabel('#')
        plt.title(f'Integral Distribution of a single event, ch {ch}, adc {adc} & FAILED TO FIT')

        # Generate PDF file name
        pdf_file_name = f'{pdf_file_path}/{data_file_name}_ADC{adc}_CH{ch}_plot_FAILED_TO_FIT.pdf'
    
        # Save plot as PDF
        plt.savefig(pdf_file_name)
        print(f"Plot saved as '{pdf_file_name}'.")

        plt.close()
        return p, c

def estimate_start_parameters(data):
    """
    Estimates the fitting starting parameters.

        Args:
    - data (array): data array.

        Returns:
    - p0 (array (3D) with mean, std and 1)
    """
    p0 = [np.mean(data), np.std(data), 1]
    return p0

def create_parm_file(file_path, file_name):
    """
    Creates the new file to save the parameter values.

        Args:
    - file_path (str): Path to the file.
    """
    # Create the destination folder if it doesn't exist
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(file_path+"/"+file_name, 'w') as file:
        # Write parameter names
        file.write("file, adc, channel, mu, sigma, A, covariance matrix\n")

def create_sigma_mean_file(file_path):
    """
    Creates the new file to save the parameter values.

        Args:
    - file_path (str): Path to the file.
    """
    with open(file_path, 'w') as file:
        # Write parameter names
        file.write("file, sigma mean, standard deviation\n")

def append_parameters_to_file(file_path, data_file_name, adc, channel, params, covariance):
    """
    Append values to the next empty line in the file.

    Args:
    - file_path (str): Path to the file.
    - data_file_name (str): name of the calibration file
    - adc (int): number of the adc
    - channel (int): number of the channel
    - params (tuple): Tuple containing parameter values (mu, sigma, A).
    - covariance (matrix): covariance of the parameters
    """
    with open(file_path, 'r+') as file:
        # Find the next empty line in the file
        for line in file:
            if not line.strip():
                break
        else:
            # If no empty line found, move cursor to the end of the file
            file.seek(0, 2)
            #file.write('\n')  # Add a new line
        
        #Write data_file_name, adc and channel to the file
        file.write(data_file_name)
        file.write(", {}, ".format(adc))
        file.write("{}, ".format(channel))

        # Write parameters to the file
        file.write("{:.6f}, {:.6f}, {:.6f}, ".format(*params))
        
        # Write covariance matrix to the file, all on one line with brackets
        file.write("[, ")
        for row in covariance:
            file.write("[, {}, ], ".format(", ".join("{:.6f}".format(entry) for entry in row)))
        file.write("]\n")

def append_parameters_to_sigma_file(file_path, data_file_name, sigma, std):
    """
    Append values to the next empty line in the file.

    Args:
    - file_path (str): Path to the file that will be filled.
    - data_file_name (str): name of the calibration file
    - sigma (int): mean of the sigma
    - std (int): mean of the std
    """
    with open(file_path, 'r+') as file:
        # Find the next empty line in the file
        for line in file:
            if not line.strip():
                break
        else:
            # If no empty line found, move cursor to the end of the file
            file.seek(0, 2)
            #file.write('\n')  # Add a new line
        
        #Write data_file_name, adc and channel to the file
        file.write(data_file_name)
        file.write(", {}, ".format(sigma))
        file.write("{}".format(std))

def create_folder(folder_path):
    """
    Create a new folder at the specified path if it doesn't already exist.

    Args:
    - folder_path (str): Path to the new folder.
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        except OSError as e:
            print(f"Failed to create folder '{folder_path}': {e}")
    else:
        print(f"Folder '{folder_path}' already exists.")

def run_fit_over_all_adc_ch(integrated_data, parm_file_name, hdf5_file_name):
    """
    Runs over all the adc and channels to do the fitting of the gaussian, creates the plots and saves them. The parameters of the fit are saved as well.

    Ags:
    - integrated_data (array): data of the integrated waveforms saved in array [Meas, ADC, Channel]
    - parm_file_name (str): name of the file to save the parameters in to
    - hdf5_file_name (str): name of the input file
    """
    #create the folder to save the plots
    plot_folder_name = parm_file_name[:-4]
    create_folder(plot_folder_name)

    # Iteration over all indices of the second and third dimension, for fitting all of them to a gaussian
    for adc in range(integrated_data.shape[1]):  # second dimension
        for ch in range(integrated_data.shape[2]):  # third dimension
            print(f'adc: {adc}')
            print(f'ch: {ch}')
            params, covariance = plot_the_int_distr_for_one_w_fit(integrated_data, hdf5_file_name, plot_folder_name, adc, ch)
            #print("Parameter: ",params)
            #print("Covariance: ",covariance)
            append_parameters_to_file(plot_folder_name+"/"+parm_file_name, hdf5_file_name, adc, ch, params, covariance)

def run_fit_over_all_adc_ch_from_json(integrated_data, parm_file_name, hdf5_file_name, json_data):
    """
    Runs over all the adc and channels to do the fitting of the gaussian, creates the plots and saves them. The parameters of the fit are saved as well.

    Ags:
    - integrated_data (array): data of the integrated waveforms saved in array [Meas, ADC, Channel]
    - parm_file_name (str): name of the file to save the parameters in to
    - hdf5_file_name (str): name of the input file
    """
    #create the folder to save the plots
    plot_folder_name = parm_file_name[:-4]
    create_folder(plot_folder_name)

    # Iteration over all indices of the second and third dimension, for fitting all of them to a gaussian
    for adc in range(integrated_data.shape[1]):  # second dimension
        for ch in range(integrated_data.shape[2]):  # third dimension
            print(f'adc: {adc}')
            print(f'ch: {ch}')
            adc_string_name = rename_adc(adc)
            #print(adc_string_name)
            if(select_channel_by_JSON_file(ch, adc_string_name, hdf5_file_name, json_data)):
                params, covariance = plot_the_int_distr_for_one_w_fit(integrated_data, hdf5_file_name, plot_folder_name, adc, ch)
                #print("Parameter: ",params)
                #print("Covariance: ",covariance)
                append_parameters_to_file(plot_folder_name+"/"+parm_file_name, hdf5_file_path, adc, ch, params, covariance)

def rename_adc(adc):
    if(adc == 0):
        adc_string = "0CD94138".lower()
        return adc_string
    elif(adc == 1):
        adc_string = "0CD8D648".lower()
        return adc_string
    elif(adc == 2):
        adc_string = "03222650B3FF".lower()
        return adc_string
    elif(adc == 3):
        adc_string = "0CD913FB".lower()
        return adc_string   
    elif(adc == 4):
        adc_string = "0CD93DB0".lower()
        return adc_string
    elif(adc == 5):
        adc_string = "032226517DFF".lower()
        return adc_string
    elif(adc == 6):
        adc_string = "0cd94149".lower()
        return adc_string   
    elif(adc == 7):
        adc_string = "032225B771FF".lower()
        return adc_string
    else:
        return " "


def run_fit_over_one_adc_ch(integrated_data, parm_file_name, hdf5_file_name, adc, ch):
    """
    Runs over one adc and one channel to do the fitting of the gaussian, creates the plots and saves them. The parameters of the fit are saved as well.

    Ags:
    - integrated_data (array): data of the integrated waveforms saved in array [Meas, ADC, Channel]
    - parm_file_name (str): name of the file to save the parameters in to
    - hdf5_file_name (str): name of the input file
    - adc (int): adc value int
    - ch (int): channel value int
    """
    plot_folder_name = parm_file_name[:-4]
    create_folder(plot_folder_name)
    params, covariance = plot_the_int_distr_for_one_w_fit(integrated_data, hdf5_file_name, plot_folder_name, adc, ch)
    #print("Parameter: ",params)
    #print("Covariance: ",covariance)
    append_parameters_to_file(plot_folder_name+"/"+parm_file_name, hdf5_file_name, adc, ch, params, covariance)

def calculate_mean_sigma_from_parm_file(parm_file_name):
    """
    Open the parameter file and calculate the mean of the values in the fourth column, skipping the first row.

    Args:
    - param_file_name (str): Name of the parameter file.

    Returns:
    - mean (float or None): Mean of the values in the fifth column if successful, else None.
    - std (float or None): Standard deviation of the values in the fourth column if successful, else None.
    """
    try:
        with open(parm_file_name, 'r') as file:
            # Skip the first row
            next(file)
            # Read values from the fourth column
            values = []
            for line in file:
                if float(line.split(', ')[4]) > 0:
                    values.append(float(line.split(', ')[4]))
            #print(values)
            # Calculate mean
            mean = sum(values) / len(values)
            # Calculate standard deviation
            std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
            return mean, std
    except (FileNotFoundError, ValueError, IndexError, ZeroDivisionError):
        print(f"Error: Unable to calculate mean from {parm_file_name}.")
        return None, None

def plot_mean_sig_values_with_errors(file_name):
    """
    Read a file, take the second column as values and the third column as errors, and plot them.

    Args:
    - file_name (str): Name of the file containing the data.
    """
    try:
        # Read data from CSV file
        with open(file_name, 'r') as file:
            next(file)  # Skip the header
            values = []
            errors = []
            x_labels = []
            for line in file:
                line_values = line.strip().split(', ')
                x_labels.append(line_values[0][23:-10])
                values.append(float(line_values[1]))
                errors.append(float(line_values[2]))
            #print(values)
            #print(errors)

        # Plot data
        plt.errorbar(range(len(values)), values, yerr=errors, fmt='o', color='b', markersize=4, capsize=5)
        plt.xticks(range(len(x_labels)), x_labels)  # Set x-axis labels
        plt.xlabel('file')
        plt.ylabel('mean value of the sigma')
        plt.title('mean values of the sigmas with errors')
        plt.grid(True)
        
        # Save plot as PDF
        plt.savefig(file_name[:-4])
        print(f"Plot saved as '{file_name}'.")

        #plt.close()
    except (FileNotFoundError, ValueError, IndexError):
        print(f"Error: Unable to read data from {file_name}")

def read_json_file(file_path):
    """
    Read data from a JSON file.
        
    Args:
    - file_path (str): The path to the JSON file.
        
    Returns:
    - data (dict): The data read from the JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not a valid JSON file.")
        return None

def select_channel_by_JSON_file(channel, adc, hdf5_file_name, json_data):
    for entry in json_data:
        if (hdf5_file_name == entry.get("calib_run").rsplit('/')[-1]):
            json_adc = entry.get("adc_serials")
            #print(json_adc)
            local_json_ch = entry.get("local_channels")
            for i in range(len(json_adc)):
                #print(type(local_json_ch[i]))
                #print(type(channel))
                #print(json_adc[i][:].lower())
                #print(adc)
                if(adc == json_adc[i][:].lower()) and (channel == int(local_json_ch[i])): 
                    print("TRUE")
                    return True
            
    return False


if __name__ == "__main__":

    json_file_path = "/home/jan/Universitat_Bern/Doktor/ADC_Viewer/2x2/sipm_resolution/2024.05.08.14.59.25.json"
    json_data = read_json_file(json_file_path)

    for entry in json_data:
        hdf5_file_path = entry.get("calib_run").rsplit('/',1)[:-1][0]+"/" #path to the *.hdf5 file
        hdf5_file_name = entry.get("calib_run").rsplit('/')[-1] #name of the *.hdf5 file
        #print(hdf5_file_name)
        
        parm_file_name=hdf5_file_name[:-5].replace(".", "_")+".csv"

        dataset_path = '/light/wvfm/data/'
        data = open_hdf5_dataset(hdf5_file_path+"/"+hdf5_file_name, dataset_path)
        """
        #plot a specific waveform:
        meas_nr = 10
        adc = 2
        ch = 20
        plot_the_wvf(data["samples"], meas_nr, adc, ch)
        """
        #integrate the data
        integrated_data = integrate_for_gate(data["samples"],80,120)

        #start saving the data in files
        create_parm_file(parm_file_name[:-4],parm_file_name)

        run_fit_over_all_adc_ch_from_json(integrated_data, parm_file_name, hdf5_file_name, json_data)
    
        m, std = calculate_mean_sigma_from_parm_file(parm_file_name[:-4]+"/"+parm_file_name)

        sig_name = json_file_path.split('/')[-1][:-5]
        sig_name = sig_name.replace(".", "_")+".csv"
        create_sigma_mean_file(sig_name)
        append_parameters_to_sigma_file(sig_name, hdf5_file_name, m, std)
        plot_mean_sig_values_with_errors(sig_name)





    #without JSON
    """
    hdf5_file_path = 'MiniRun5_1E19_RHC.flow.0001023.FLOW.hdf5'

    parm_file_name=hdf5_file_path[:-5]+".csv"

    #print_hdf5_structure(hdf5_file_path)
    dataset_path = '/light/wvfm/data/'
    data = open_hdf5_dataset(hdf5_file_path, dataset_path)
    #print("Content of the dataset:")
    #print(data["samples"].shape)
    """
    """
    #plot a specific waveform:
    meas_nr = 10
    adc = 2
    ch = 20
    plot_the_wvf(data["samples"], meas_nr, adc, ch)
    """
    """
    #integrate the data
    integrated_data = integrate_for_gate(data["samples"],80,120)
    #take the mean of the data
    #mean_data = mean_for_integral(integrated_data)
    """
    """
    #plot the integral distribution for one channel
    adc = 2
    ch = 20
    plot_the_int_distr_for_one(integrated_data, adc, ch)
    """
    #start saving the data in files
    #create_parm_file(parm_file_name[:-4],parm_file_name)

    #run_fit_over_one_adc_ch(integrated_data, parm_file_name, hdf5_file_path, adc, ch)

    #run_fit_over_all_adc_ch(integrated_data, parm_file_name, hdf5_file_path)
    """
    m, std = calculate_mean_sigma_from_parm_file(parm_file_name[:-4]+"/"+parm_file_name)
    #print(m)
    #print(std)
    sig_name = "test_resoulution_sig1.csv"
    create_sigma_mean_file(sig_name)
    append_parameters_to_sigma_file(sig_name, hdf5_file_path, m, std)
    plot_mean_sig_values_with_errors(sig_name)

    """
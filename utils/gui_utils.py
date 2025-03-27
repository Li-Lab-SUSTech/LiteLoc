# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 15:10:12 2025

@author: Spike RX Wang
"""

import tkinter as tk
from tkinter import filedialog, messagebox

def show_confirming_string(Dict, ini = True):
    show_str = '' 
    for i, (key,value) in enumerate(Dict.items(), 1):
        if not isinstance(value, dict):
            show_str += f"{key} : {value}"
            if not ini:
                show_str += '\n'
        else:
            show_str += f"--{key} :\n"
            show_str += show_confirming_string(value, False)
            
    return show_str

def Camera_GUI():

    def on_camera_select(option):
        if camera_vars[option].get() == 1:
            for i in range(2):
                if i != option:
                    camera_vars[i].set(0)
        else:
            camera_vars[option].set(0)

    def get_parameters():
        option_camera = None
        for i in range(2):
            if camera_vars[i].get() == 1:
                option_camera = i + 1
                option_camera = 'sCMOS' if option_camera == 1 else 'EMCCD'
                break
        
        value_dict = {'camera': option_camera}
        for key, value in entry_dict.items():
            value_dict[key] = float(value.get().strip()) if value.get().strip() else None
            
        return value_dict
    
    
    def show_result_window(result):
        result_window = tk.Toplevel(root)
        result_window.title("Setting Confirming...")
    
        show_str = "Settings:\n" + show_confirming_string(parameters)
        # show_str = "Settings:\n"
        # for i, (key, value) in enumerate(parameters.items(), 1):
        #     show_str += f"{key} : {value}"#f"自定义显示内容:\n{result}"
        #     if i != len(parameters):
        #         show_str += "\n"
        result_label = tk.Label(result_window, text=show_str, justify=tk.LEFT, anchor=tk.W)
        result_label.pack(padx=20, pady=20)
    
        close_button = tk.Button(result_window, text="Confirm", command=root.destroy)
        close_button.pack(side=tk.RIGHT, padx=10, pady=10)
    
        retry_button = tk.Button(result_window, text="Redo", command=result_window.destroy)
        retry_button.pack(side=tk.RIGHT, padx=10, pady=10)
    
    def submit():
        global parameters
        parameters = get_parameters()
        show_result_window(parameters)

    # Main Window
    root = tk.Tk()
    root.title("Camera Parameter Setting")
    root.geometry("600x400")
    
    # Frame setting
    main_frame = tk.Frame(root)
    main_frame.pack(padx=20, pady=20)

    # camera
    camera_frame = tk.Frame(main_frame)
    camera_frame.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
    
    camera_label = tk.Label(camera_frame, text="camera :", anchor=tk.W, width=15)
    camera_label.grid(row=0, column=0, padx=5, sticky=tk.W)
    
    camera_vars = [tk.IntVar(value=0) for _ in range(2)]
    camera_vars[0].set(1)
    camera_button1 = tk.Checkbutton(camera_frame, text="sCMOS", variable=camera_vars[0], command=lambda: on_camera_select(0))
    camera_button1.grid(row=0, column=1, padx=5, sticky=tk.W)
    
    camera_button2 = tk.Checkbutton(camera_frame, text="EMCCD", variable=camera_vars[1], command=lambda: on_camera_select(1))
    camera_button2.grid(row=0, column=2, padx=5, sticky=tk.W)
    
    # Input bar
    item_dict = {
        'em_gain': 1.0, 
        'surp_p': 0.5, 
        'qe': 0.95, 
        'spurious_c': 0.002, 
        'sig_read': 1.535,
        'e_per_adu': 0.7471,
        'baseline': 100.0
        }
    
    frame_dict = {}
    label_dict = {}
    entry_dict = {}
    
    for idx, (key, value) in enumerate(item_dict.items(), 1):
        frame_dict[key] = tk.Frame(main_frame)
        frame_dict[key].grid(row = idx, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        label_dict[key] = tk.Label(frame_dict[key], text="{} :".format(key), anchor=tk.W, width=15)
        label_dict[key].grid(row=0, column=0, padx=5, sticky=tk.W)
        
        entry_dict[key] = tk.Entry(frame_dict[key], width=30)
        entry_dict[key].grid(row=0, column=1, padx=5)
        entry_dict[key].insert(0, "{}".format(value))
    
    # Submit
    submit_button = tk.Button(main_frame, text="Submit", command=submit)
    submit_button.grid(row=8, column=0, columnspan=3, pady=20)
    
    # Run main loop
    root.mainloop()

    return parameters

def Training_GUI():
# def window_one():

    def on_noise_select(option):
        if noise_vars[option].get() == 1:
            for i in range(2):
                if i != option:
                    noise_vars[i].set(0)
        else:
            noise_vars[option].set(0)
    
    def select_file():
        file_path = filedialog.askopenfilename()
        if file_path:
            # file_entry.delete(0, tk.END)
            # file_entry.insert(0, file_path)
            entry_dict['infer'].delete(0, tk.END)
            entry_dict['infer'].insert(0, file_path)
    
    def select_folder():
        folder_path = filedialog.askdirectory()
        if folder_path:
            entry_dict['result_path'].delete(0, tk.END)
            entry_dict['result_path'].insert(0, folder_path)
    
    def get_parameters():
        
        value_dict = {}
        
        type_dict = {
            'max_epoch': int,
            'eval_iteration': int,
            'batch_size': int,
            'valid_frame_num': int,
            'em_per_frame': int,
            'train_size': int,
            'photon_range': int,
            'result_path': str,
            'infer_data': str,
            'bg': float,
            'perline_noise': bool,
            'pn_factor': float,
            'pn_res': int,
            'factor': float,
            'offset': float,
            'model_init': str,
            'project_name': str
            }
               
    
        for key, value in entry_dict.items():
            # print(key, 'here')
            if key == 'perline_noise':
                
                value_dict[key] = None
                for i in range(2):
                    if noise_vars[i].get() == 1:
                        value_dict[key] = (i + 1 == 1)
                        break
            elif not isinstance(value, dict):
                value_dict[key] = type_dict[key](value.get().strip()) if value.get().strip() else None
            else:
                value_dict[key] = [type_dict[key](v.get().strip()) if v.get().strip() else None for k, v in value.items()]
                
 
        return value_dict
    
    def show_result_window(result):
        result_window = tk.Toplevel(root)
        result_window.title("Setting confriming...")
    
        show_str = "Settings:\n" + show_confirming_string(parameters)
        # show_str = "Settings:\n"
        # for i, (key, value) in enumerate(parameters.items(), 1):
        #     show_str += f"{key} : {value}"#f"自定义显示内容:\n{result}"
        #     if i != len(parameters):
        #         show_str += "\n"
        result_label = tk.Label(result_window, text=show_str, justify=tk.LEFT, anchor=tk.W)
        result_label.pack(padx=20, pady=20)
    
        close_button = tk.Button(result_window, text="Confirm", command=root.destroy)
        close_button.pack(side=tk.RIGHT, padx=10, pady=10)
    
        retry_button = tk.Button(result_window, text="Redo", command=result_window.destroy)
        retry_button.pack(side=tk.RIGHT, padx=10, pady=10)
    
    def submit():
        global parameters
        parameters = get_parameters()
        show_result_window(parameters)

    # Main Window
    root = tk.Tk()
    root.title("Training Parameter Setting")
    root.geometry("600x800")
    
    # Fram setting
    main_frame = tk.Frame(root)
    main_frame.pack(padx=20, pady=20)
    
    item_dict = {
        'max_epoch': 50,
        'eval_iteration': 500,
        'batch_size': 16,
        'valid_frame_num': 100,
        'em_per_frame': 10,
        'train_size': [64, 64],
        'photon_range': [4000, 40000],
        'result_path': None,
        'infer_data': None,
        'bg': None,
        'perline_noise': True,
        'pn_factor': 0.2,
        'pn_res': 64,
        'factor': None,
        'offset': None,
        'model_init': None,
        'project_name': 'LiteLoc-main'
        }
    
    list_value_dict = {
        'train_size': ['Height', 'Width'],
        'photon_range': ['min', 'max'],
        }

    frame_dict = {}
    label_dict = {}
    entry_dict = {}
    
    row_number = -1
    for key, value in item_dict.items():
        # if isinstance(value, int) or isinstance(value, float) or (value is None) or (key == 'project_name'):
            
        row_number += 1
        
        if key == 'perline_noise':
            
            # perline noise selection
            frame_dict[key] = tk.Frame(main_frame)
            frame_dict[key].grid(row=row_number, column=0, columnspan=3, sticky=tk.W, pady=5)
            
            label_dict[key] = tk.Label(frame_dict[key], text="{}：".format(key), anchor=tk.W, width=15)
            label_dict[key].grid(row=0, column=0, padx=5, sticky=tk.W)
            
            noise_vars = [tk.IntVar(value=0) for _ in range(2)]
            noise_vars[0].set(1)
            noise_button1 = tk.Checkbutton(frame_dict[key], text="True", variable=noise_vars[0], command=lambda: on_noise_select(0))
            noise_button1.grid(row=0, column=1, padx=5, sticky=tk.W)
            
            noise_button2 = tk.Checkbutton(frame_dict[key], text="False", variable=noise_vars[1], command=lambda: on_noise_select(1))
            noise_button2.grid(row=0, column=2, padx=5, sticky=tk.W)
            
            entry_dict[key] = None
                        
            
        elif not isinstance(value, list):
            # row_number += 1
            frame_dict[key] = tk.Frame(main_frame)
            frame_dict[key].grid(row=row_number, column=0, columnspan=3, sticky=tk.W, pady=5)
            
            label_dict[key] = tk.Label(frame_dict[key], text="{} :".format(key))
            label_dict[key].grid(row=0, column=0, padx=5, sticky=tk.W)
            
            entry_dict[key] = tk.Entry(frame_dict[key], width = 30)
            entry_dict[key].grid(row=0, column=1, padx=5)
            if value is not None:
                entry_dict[key].insert(0, value)
                
            if key == 'infer':
                file_button = tk.Button(frame_dict[key], text="Open", command=select_file)
                file_button.grid(row=0, column=2, padx=5)
                
            elif key =='result_path':
                folder_button = tk.Button(frame_dict[key], text="Open", command=select_folder)
                folder_button.grid(row=0, column=2, padx=5)
                
                
        else:
            # row_number += 1
            frame_dict[key] = {}
            frame_dict[key][0] = tk.Frame(main_frame)
            frame_dict[key][0].grid(row=row_number, column=0, sticky=tk.W, pady=5)
            
            label_dict[key] = {}
            label_dict[key][0] = tk.Label(frame_dict[key][0], text="{} ：".format(key), anchor=tk.W)
            label_dict[key][0].grid(row=0, column=0, padx=5, sticky=tk.W)
            
            # print(key)
            entry_dict[key] = {}
            # print(key)
            
            row_number += 1
            
            for idx, list_value in enumerate(value, 1):
                frame_dict[key][idx] = tk.Frame(main_frame)
                frame_dict[key][idx].grid(row=row_number, column=idx - 1, sticky=tk.W, pady=5)
                
                label_dict[key][idx] = tk.Label(frame_dict[key][idx], text="{}：".format(list_value_dict[key][idx-1]), anchor=tk.W)
                label_dict[key][idx].grid(row=0, column=0, padx=5, sticky=tk.W)
                
                entry_dict[key][idx] = tk.Entry(frame_dict[key][idx], width=10)
                entry_dict[key][idx].grid(row=0, column=1, padx=5)
                entry_dict[key][idx].insert(0, value[idx - 1])
    
    # submission
    row_number += 1
    submit_button = tk.Button(main_frame, text="Submit", command=submit)
    submit_button.grid(row=row_number, column=0, columnspan=3, pady=20)
    
    # main loop
    root.mainloop()

    return parameters

def PSF_GUI():
    
    def on_method_select(option):
        if method_vars[option].get() == 1:
            for i in range(2):
                if i != option:
                    method_vars[i].set(0)
        else:
            method_vars[option].set(0)
            
    def on_device_select(option):
        if device_vars[option].get() == 1:
            for i in range(2):
                if i != option:
                    device_vars[i].set(0)
        else:
            device_vars[option].set(0)
    
    # def select_file_cal():
    #     file_path = filedialog.askopenfilename()
    #     if file_path:
    #         # file_entry.delete(0, tk.END)
    #         # file_entry.insert(0, file_path)
    #         entry_dict['spline_psf']['calibration_file'].delete(0, tk.END)
    #         entry_dict['spline_psf']['calibration_file'].insert(0, file_path)
            
    # def select_file_zff():
    #     file_path = filedialog.askopenfilename()
    #     if file_path:
    #         # file_entry.delete(0, tk.END)
    #         # file_entry.insert(0, file_path)
    #         entry_dict['vector_psf']['zernikefit_file'].delete(0, tk.END)
    #         entry_dict['spline_psf']['zernikefit_file'].insert(0, file_path)
            
    # def select_file_zfm():
    #     file_path = filedialog.askopenfilename()
    #     if file_path:
    #         # file_entry.delete(0, tk.END)
    #         # file_entry.insert(0, file_path)
    #         entry_dict['vector_psf']['calibration_map'].delete(0, tk.END)
    #         entry_dict['vector_psf']['calibration_map'].insert(0, file_path)
            
    class Select_File():
        def __init__(self, name1, name2):
            self.name1 = name1
            self.name2 = name2
        def __call__(self):
            file_path = filedialog.askopenfilename()
            if file_path:
                entry_dict[self.name1][self.name2].delete(0, tk.END)
                entry_dict[self.name1][self.name2].insert(0, file_path)
    
    def get_parameters():
        
        value_dict = {}
        
        type_dict = {
            'z_scale': float,
            'simulate_method': str,
            # 'spline_psf': {
                'calibration_file': None,
                'psf_extent': float,
                'device_simulation': str,
                # },
            # 'vector_psf': {
            # 'row1': {
                'objstage0': float,
                'zemit0': float,
                    # },
                'zernikefit_file': str,
                # 'pixelSize': {           #[110, 110], # pixelX, pixelY
                'pixelSizeX': int,
                'pixelSizeY': int,
                # },
                # 'row4': {
                'psfSizeX': int,
                'NA': float,
                'wavelength': float,
                # },
                # 'row5': {
                'refmed': float,
                'refcov': float,
                'refimm': float,
                    # },
                'zernikefit_map': str,
                # 'row7':{
                'psfrescale': float,
                'Npupil': int
                    # }
                
                # }
            }
                     
    
        for key, value in entry_dict.items():
            if key == 'psf_extent':
                # print(value)
                assert len(value) == 4
                value_dict[key] = [
                    [
                        type_dict[key](value[j + i * 2 + 1].get().strip()) if value[j + i * 2 + 1].get().strip() else None for j in range(2)
                        ] for i in range(2)
                    ]
                value_dict[key].append (None)
            elif key in ['simulate_method', 'device_simulation']:
            # # print(key, 'here')
            # if key == 'simulate_method':
                
                value_dict[key] = None
                for i in range(2):
                    if method_vars[i].get() == 1:
                        if key == 'simulate_method':
                            value_dict[key] = 'spline' if (i + 1 == 1) else 'vector'
                        else:
                            value_dict[key] = 'cuda' if (i + 1 == 1) else 'cpu'
                        break
                    
            else:
                value_dict[key] = type_dict[key](value.get().strip()) if value.get().strip() else None
                    
            # elif not isinstance(value, dict):
            #     value_dict[key] = type_dict[key](value.get().strip()) if value.get().strip() else None
            # else:
            #     value_dict[key] = [type_dict[key](v.get().strip()) if v.get().strip() else None for k, v in value.items()]
            
        result_dict = {}
        for k in ['z_scale', 'simulate_method']:
            result_dict[k] = value_dict[k]
        result_dict['spline_psf'] = {}
        for k in ['calibration_file', 'psf_extent', 'device_simulation']:
            result_dict['spline_psf'][k] = value_dict[k]
        result_dict['vector_psf'] = {}
        for k in value_dict:
            if k not in ['z_scale', 'simulate_method', 'calibration_file', 'psf_extent', 'device_simulation']:
                result_dict['vector_psf'][k] = value_dict[k]
 
        return result_dict
    
    def show_result_window(result):
        result_window = tk.Toplevel(root)
        result_window.title("Setting confriming...")
    
        show_str = "Settings:\n" + show_confirming_string(parameters)
        # show_str = "Settings:\n"
        # for i, (key, value) in enumerate(parameters.items(), 1):
        #     show_str += f"{key} : {value}"#f"自定义显示内容:\n{result}"
        #     if i != len(parameters):
        #         show_str += "\n"
        result_label = tk.Label(result_window, text=show_str, justify=tk.LEFT, anchor=tk.W)
        result_label.pack(padx=20, pady=20)
    
        close_button = tk.Button(result_window, text="Confirm", command=root.destroy)
        close_button.pack(side=tk.RIGHT, padx=10, pady=10)
    
        retry_button = tk.Button(result_window, text="Redo", command=result_window.destroy)
        retry_button.pack(side=tk.RIGHT, padx=10, pady=10)
    
    
    
    def submit():
        global parameters
        parameters = get_parameters()
        show_result_window(parameters)

    # Main Window
    root = tk.Tk()
    root.title("PSF Parameter Setting")
    root.geometry("800x800")
    
    # Fram setting
    main_frame = tk.Frame(root)
    main_frame.pack(padx=20, pady=20)
    
    item_dict = {
        'z_scale': 700,
        'simulate_method': 'vector',
        'spline_psf': {
            'calibration_file': None,
            'psf_extent': [[-0.5, 63.5], [-0.5, 63.5], None],
            'device_simulation': 'cuda',
            },
        'vector_psf': {
            'row1': {
                'objstage0': -500,
                'zemit0': None,
                },
            'zernikefit_file': None,
            'pixelSize': {           #[110, 110], # pixelX, pixelY
                'pixelSizeX': 110,
                'pixelSizeY': 110,
            },
            'row4': {
                'psfSizeX': 51,
                'NA': 1.5,
                'wavelength': 680,
            },
            'row5': {
                'refmed': 1.406,
                'refcov': 1.524,
                'refimm': 1.518,
                },
            'zernikefit_map': None,
            'row7':{
                'psfrescale': 0.5,
                'Npupil': 64
                }
            
        }
        }
    

    frame_dict = {}
    label_dict = {}
    entry_dict = {}
    
    row_number = -1
    for key, value in item_dict.items():
        # if isinstance(value, int) or isinstance(value, float) or (value is None) or (key == 'project_name'):
            
        
        
        if key == 'simulate_method':
            
            row_number += 1
            
            # perline noise selection
            frame_dict[key] = tk.Frame(main_frame)
            frame_dict[key].grid(row=row_number, column=0, columnspan=3, sticky=tk.W, pady=5)
            
            label_dict[key] = tk.Label(frame_dict[key], text="{}：".format(key), anchor=tk.W, width=15)
            label_dict[key].grid(row=0, column=0, padx=5, sticky=tk.W)
            
            method_vars = [tk.IntVar(value=0) for _ in range(2)]
            method_vars[0].set(1)
            method_button1 = tk.Checkbutton(frame_dict[key], text="spline", variable=method_vars[0], command=lambda: on_method_select(0))
            method_button1.grid(row=0, column=1, padx=5, sticky=tk.W)
            
            method_button2 = tk.Checkbutton(frame_dict[key], text="vector", variable=method_vars[1], command=lambda: on_method_select(1))
            method_button2.grid(row=0, column=2, padx=5, sticky=tk.W)
            
            entry_dict[key] = None
                          
            
        elif key == 'z_scale':
            row_number += 1
            frame_dict[key] = tk.Frame(main_frame)
            frame_dict[key].grid(row=row_number, column=0, columnspan=3, sticky=tk.W, pady=5)
            
            label_dict[key] = tk.Label(frame_dict[key], text="{} :".format(key))
            label_dict[key].grid(row=0, column=0, padx=5, sticky=tk.W)
            
            entry_dict[key] = tk.Entry(frame_dict[key], width = 30)
            entry_dict[key].grid(row=0, column=1, padx=5)
            entry_dict[key].insert(0, value)

                
        else:
            # row_number += 1
            file_button = {}
            for kk, vv in value.items():
                if kk in ['calibration_file', 'zernikefit_file', 'zernikefit_map']:
                    row_number += 1
                    
                    frame_dict[kk] = tk.Frame(main_frame)
                    frame_dict[kk].grid(row=row_number, column=0, columnspan=3, sticky=tk.W, pady=5)
                    
                    label_dict[kk] = tk.Label(frame_dict[kk], text="{} :".format(kk))
                    label_dict[kk].grid(row=0, column=0, padx=5, sticky=tk.W)
                    
                    entry_dict[kk] = tk.Entry(frame_dict[kk], width = 30)
                    entry_dict[kk].grid(row=0, column=1, padx=5)
                    if vv is not None:
                        entry_dict[kk].insert(0, vv)
                        
                    file_button[kk] = tk.Button(frame_dict[kk], text="Open", command=Select_File(key, kk))
                    file_button[kk].grid(row=0, column=2, padx=5)
                    # elif key == 'spline_psf':
                    #     file_button = tk.Button(frame_dict[key], text="Open", command=select_file)
                    #     file_button.grid(row=0, column=2, padx=5)
                    # elif key =='result_path':
                    #     folder_button = tk.Button(frame_dict[key], text="Open", command=select_folder)
                    #     folder_button.grid(row=0, column=2, padx=5)
                    
                elif kk == 'device_simulation':

                    row_number += 1
                    
                    # perline noise selection
                    frame_dict[kk] = tk.Frame(main_frame)
                    frame_dict[kk].grid(row=row_number, column=0, columnspan=3, sticky=tk.W, pady=5)
                    
                    label_dict[kk] = tk.Label(frame_dict[kk], text="{}：".format(key), anchor=tk.W, width=15)
                    label_dict[kk].grid(row=0, column=0, padx=5, sticky=tk.W)
                    
                    device_vars = [tk.IntVar(value=0) for _ in range(2)]
                    device_vars[0].set(1)
                    device_button1 = tk.Checkbutton(frame_dict[kk], text="cuda", variable=device_vars[0], command=lambda: on_device_select(0))
                    device_button1.grid(row=0, column=1, padx=5, sticky=tk.W)
                    
                    device_button2 = tk.Checkbutton(frame_dict[kk], text="cpu", variable=device_vars[1], command=lambda: on_device_select(1))
                    device_button2.grid(row=0, column=2, padx=5, sticky=tk.W)
                    
                    entry_dict[kk] = None
                    
                elif kk == 'psf_extent':
                    
                    row_number += 1
                    
                    frame_dict[kk] = {}
                    frame_dict[kk][0] = tk.Frame(main_frame)
                    frame_dict[kk][0].grid(row=row_number, column=0, sticky=tk.W, pady=5)
                    
                    label_dict[kk] = {}
                    label_dict[kk][0] = tk.Label(frame_dict[kk][0], text="{} ：".format(kk), anchor=tk.W) 
                    label_dict[kk][0].grid(row=0, column=0, padx=3, sticky=tk.W)
                    
                    entry_dict[kk] = {}
                    
                    row_number += 1
                    
                    list_values = ['h_min', 'h_max', 'w_min', 'w_max']
                    for idx, list_value in enumerate(list_values, 1):
                        frame_dict[kk][idx] = tk.Frame(main_frame)
                        frame_dict[kk][idx].grid(row=row_number, column=idx - 1, sticky=tk.W, pady=5)
                        
                        label_dict[kk][idx] = tk.Label(frame_dict[kk][idx], text="{} ：".format(list_value), anchor=tk.W)
                        label_dict[kk][idx].grid(row=0, column=0, padx=5, sticky=tk.W)
                        
                        entry_dict[kk][idx] = tk.Entry(frame_dict[kk][idx], width=6)
                        entry_dict[kk][idx].grid(row=0, column=1, padx=0)
                        entry_dict[kk][idx].insert(0, vv[(idx - 1) // 2][(idx - 1) % 2])
                
                else:
                    # print(kk)
                    assert isinstance(vv, dict)
                    row_number += 1
                    
                    for idx, (kkk, vvv) in enumerate(vv.items()):
                        
                        
                        
                        frame_dict[kkk] = tk.Frame(main_frame)
                        frame_dict[kkk].grid(row=row_number, column=idx, sticky=tk.W, pady=5)
                        
                        label_dict[kkk] = tk.Label(frame_dict[kkk], text="{} :".format(kkk))
                        label_dict[kkk].grid(row=0, column=0, padx=5, sticky=tk.W)
                        
                        entry_dict[kkk] = tk.Entry(frame_dict[kkk], width = 15 if len(vv) <3 else 10)
                        entry_dict[kkk].grid(row=0, column=1, padx=5)
                        if vvv is not None:
                            entry_dict[kkk].insert(0, vvv)
                    
    # submission
    row_number += 1
    submit_button = tk.Button(main_frame, text="Submit", command=submit)
    submit_button.grid(row=row_number, column=0, columnspan=3, pady=20)
    
    # main loop
    root.mainloop()
    
    return parameters

def Infer_GUI():

    def on_gpu_select(option):
        if gpu_vars[option].get() == 1:
            for i in range(2):
                if i != option:
                    gpu_vars[i].set(0)
        else:
            gpu_vars[option].set(0)
             
    class Select_File():
        def __init__(self, name1):
            self.name1 = name1
        def __call__(self):
            file_path = filedialog.askopenfilename()
            if file_path:
                entry_dict[self.name1].delete(0, tk.END)
                entry_dict[self.name1].insert(0, file_path)

    def get_parameters():
        
        value_dict = {}
        
        for key, value in entry_dict.items():
            # print(key, 'here')
            if key == 'multi_gpu':
                value_dict[key] = None
                for i in range(2):
                    if gpu_vars[i].get() == 1:
                        value_dict[key] = (i + 1 == 1)
                        break
            else:
                value_dict[key] = (str(value.get().strip()) if 'path' in key else int(value.get().strip())) if value.get().strip() else None
        
        result_dict = {'Loc_Model':{}, 'Multi_Process':{}}
        for key, value in value_dict.items():
            result_dict['Loc_Model' if key == 'model_path' else 'Multi_Process'][key] = value
            
        return result_dict
                
    def show_result_window(result):
        result_window = tk.Toplevel(root)
        result_window.title("Setting Confirming...")
    
        show_str = "Settings:\n" + show_confirming_string(parameters)    
        # show_str = "Settings:\n"
        # for i, (key, value) in enumerate(parameters.items(), 1):
        #     show_str += f"{key} : {value}"#f"自定义显示内容:\n{result}"
        #     if i != len(parameters):
        #         show_str += "\n"
        result_label = tk.Label(result_window, text=show_str, justify=tk.LEFT, anchor=tk.W)
        result_label.pack(padx=20, pady=20)
    
        close_button = tk.Button(result_window, text="Confirm", command=root.destroy)
        close_button.pack(side=tk.RIGHT, padx=10, pady=10)
    
        retry_button = tk.Button(result_window, text="Redo", command=result_window.destroy)
        retry_button.pack(side=tk.RIGHT, padx=10, pady=10)
    
    def submit():
        global parameters
        parameters = get_parameters()
        show_result_window(parameters)
    
    # Main Window
    root = tk.Tk()
    root.title("Inference Parameter Setting")
    root.geometry("600x450")
    
    # Frame setting
    main_frame = tk.Frame(root)
    main_frame.pack(padx=20, pady=20)
    
    item_dict = {
        'Loc_Model': {
            'model_path': None         
            },
        'Multi_Process':{
            'image_path': None, 
            'save_path': None, 
            'time_block_gb': 1, 
            'batch_size': 30,
            'over_cut': 8,
            'multi_gpu': True,
            'num_producers': 1
            }
        }
    
    frame_dict = {}
    label_dict = {}
    entry_dict = {}
    
    row_number = -1     
    
    for Key, Value in item_dict.items():
        
       row_number += 1
       
       frame_dict[Key] = tk.Frame(main_frame)
       frame_dict[Key].grid(row = row_number, column=0, columnspan=3, sticky=tk.W, pady=5)
       
       label_dict[Key] = tk.Label(frame_dict[Key], text="{} :".format(Key), anchor=tk.W, width=15)
       label_dict[Key].grid(row=0, column=0, padx=5, sticky=tk.W)
       
       for key, value in Value.items():
           
           row_number += 1
            
           frame_dict[key] = tk.Frame(main_frame)
           frame_dict[key].grid(row = row_number, column=0, columnspan=3, sticky=tk.W, pady=5)
           
           label_dict[key] = tk.Label(frame_dict[key], text="{} :".format(key), anchor=tk.W, width=15)
           label_dict[key].grid(row=0, column=0, padx=5, sticky=tk.W)
           
           if key =='multi_gpu':
               gpu_vars = [tk.IntVar(value=0) for _ in range(2)]
               gpu_vars[0].set(1)
               gpu_button1 = tk.Checkbutton(frame_dict[key], text="True", variable=gpu_vars[0], command=lambda: on_gpu_select(0))
               gpu_button1.grid(row=0, column=1, padx=5, sticky=tk.W)
               
               gpu_button2 = tk.Checkbutton(frame_dict[key], text="False", variable=gpu_vars[1], command=lambda: on_gpu_select(1))
               gpu_button2.grid(row=0, column=2, padx=5, sticky=tk.W)
               
               entry_dict[key] = None
           else:
               entry_dict[key] = tk.Entry(frame_dict[key], width=30 if 'path' in key else 15)
               entry_dict[key].grid(row=0, column=1, padx=5)
               if value is not None:
                        entry_dict[key].insert(0, "{}".format(value))
                        
               if 'path' in key:
                   file_button = tk.Button(frame_dict[key], text="Open", command=Select_File(key))
                   file_button.grid(row=0, column=2, padx=5)
    
    # Submit
    submit_button = tk.Button(main_frame, text="Submit", command=submit)
    submit_button.grid(row=row_number + 1, column=0, columnspan=3, pady=20)
    
    # Run main loop
    root.mainloop()
    
    return parameters   
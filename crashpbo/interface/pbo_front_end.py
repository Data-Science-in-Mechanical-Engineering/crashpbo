import streamlit as st
import numpy as np
import random
import plotly.graph_objects as go
import time as time
import json
import os
from crashpbo.utils.front_end_utils import *
import pathlib

def main():
    abs_path = pathlib.Path.cwd()
    optimization_config_path    = os.path.join(abs_path, "optimization_config.json")
    
    if 'optimization_config' not in st.session_state:
        if os.path.exists(optimization_config_path): 
            with open(optimization_config_path) as json_file:
                optimization_config_data = json.load(json_file)
                print(optimization_config_data)
            st.session_state.optimization_config = optimization_config_data 
        else:
            st.write(optimization_config_path + " not found. Has back end started?")
            time.sleep(1)
            st.rerun()
    
    if 'paths' not in st.session_state:
         st.session_state.paths = {}

    st.session_state.paths["db_path"]             =           st.session_state.optimization_config["paths"]["db_path"]
    st.session_state.paths["experiments_db_path"] =           st.session_state.optimization_config["paths"]["experiments_db_path"]
    st.session_state.paths["db_path_blocked"]     =           st.session_state.optimization_config["paths"]["db_path_blocked"]
    st.session_state.paths["experiments_db_path_blocked"] =   st.session_state.optimization_config["paths"]["experiments_db_path_blocked"]
    st.session_state.paths["stats_path"]          = os.path.join(st.session_state.optimization_config["Folder Name"], "optimizer_stats.json")
    st.session_state.paths["stats_path_blocked"] = os.path.join(st.session_state.optimization_config["Folder Name"], "optimizer_stats.json.blocked")
    st.session_state.paths["problem_def_path"]            = os.path.join(st.session_state.optimization_config["Folder Name"], "problem_def.json") 
    st.session_state.paths["opt_active_cmd_path"]         = os.path.join(st.session_state.optimization_config["Folder Name"], "opt_active")
    st.session_state.paths["rerun_right_cmd_path"]       = os.path.join(st.session_state.optimization_config["Folder Name"], "rerun_right") 
    st.session_state.paths["rerun_left_cmd_path"]       = os.path.join(st.session_state.optimization_config["Folder Name"], "rerun_left") 

    st.set_page_config(layout="wide")

    # Simulating a simple Bayesian Optimization with preferential feedback
    st.title("Preferential Bayesian Optimization (PBO) with Visualization")

    st.write("""
    This is a demo for Preferential Bayesian Optimization. Two parameter sets will be shown to you, along with their respective plots. 
    Please choose which one you prefer, and the app will refine its recommendation over time.
    """)

    if os.path.exists(st.session_state.paths["opt_active_cmd_path"]):
        during_optimization()
    else:
        before_optimization()

def generate_parameter_fields(default_param):
    
    parameters = []
    

    for i in range(len(default_param)):
        st.write(f"### Parameter {i + 1}: " + default_param[i]["name"])
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            lower_bound = st.number_input(f"Lower Bound ", key=f"lower_bound_{i}",step = None,value = default_param[i]["lower_bound"])
        
        with col2:
            upper_bound = st.number_input(f"Upper Bound ", key=f"upper_bound_{i}",step = None,value = default_param[i]["upper_bound"])
        
        with col3:
            initial_value_first = st.number_input(f"First Initial Value ", key=f"first_initial_value_{i}",step = None,value  = default_param[i]["init_first"])
        with col4:
            initial_value_second = st.number_input(f"Second Initial Value ", key=f"second_initial_value_{i}",step = None,value  = default_param[i]["init_second"])
    
        
        # Store the parameter details in a dictionary
        param_details = {
            "name": default_param[i]["name"],
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "init_first": initial_value_first,
            "init_second": initial_value_second
        }
        
        parameters.append(param_details)
    return parameters

def before_optimization():
    if os.path.exists(st.session_state.paths["problem_def_path"]): 
        with open(st.session_state.paths["problem_def_path"]) as json_data:
            d = json.load(json_data)
            json_data.close()
            print(d)
            st.session_state.problem_definition = d 
    
    
    # Generate input fields based on the selected number of parameters
        params = generate_parameter_fields(st.session_state.problem_definition["Parameters"])
        placeholder = st.empty()
        run_optimization = placeholder.button("Start Optimization",key = ["run_optimization_button"])
        if params:
            st.write("### Parameters Summary")
            for param in params:
                st.write(param) 
        if run_optimization:
            st.session_state.problem_definition["Parameters"] = params
            with open(st.session_state.paths["problem_def_path"]  , 'w') as json_file:
                print("dumping...")
                json.dump(st.session_state.problem_definition, json_file)
                print("sucesfully dumped")  
              
            open(st.session_state.paths["opt_active_cmd_path"], 'a').close()
            placeholder.empty()
            st.rerun()
    else:
        st.write("No problem definition found at " + st.session_state.paths["problem_def_path"] + " make sure that the back end has started correctly"  )
        time.sleep(2)
        st.rerun(2)
        
def during_optimization():
    succesfull_read = False
    write_db = False 

    # Step 2: Initialize the parameters
    if 'db_data' not in st.session_state:
         st.session_state.db_data = []

    if 'write_db' not in st.session_state:
        st.session_state.write_db = False

    if 'optimizer_stats' not in st.session_state:
        st.session_state.optimizer_stats = []

    if 'wait_for_experiment_repetition' not in st.session_state:
        st.session_state.wait_for_experiment_repetition = False

    # define button callbacks:

    def prefer_set_1_callback(decision_ind):
        st.session_state.db_data[decision_ind]["Decision"] = "1 > 2"
        st.session_state.write_db = True

    def prefer_set_2_callback(decision_ind):
        st.session_state.db_data[decision_ind]["Decision"] = "2 > 1"
        st.session_state.write_db = True

    def repeat_1_callback(decision_ind):
        open(st.session_state.paths["rerun_left_cmd_path"], 'a').close()
        st.session_state.wait_for_experiment_repetition = True

    def repeat_2_callback(decision_ind):
        open(st.session_state.paths["rerun_right_cmd_path"], 'a').close()
        st.session_state.wait_for_experiment_repetition = True

    def set_1_crashed_callback(decision_ind):
        st.session_state.db_data[decision_ind]["Decision"] = "1 crashed"
        st.session_state.write_db = True
        print("choice: set 1 crashed")

    def set_2_crashed_callback(decision_ind):
        st.session_state.db_data[decision_ind]["Decision"] = "2 crashed"
        st.session_state.write_db = True
        print("choice: Set 2 crashed")    

    def both_sets_crashed_callback(decision_ind):
        st.session_state.db_data[decision_ind]["Decision"] = "both crashed"
        st.session_state.write_db = True
        print("choice: Both sets crashed")

    def delete_callback(decision_ind):
        print("deleting {decision_ind}")
        st.session_state.db_data.pop(decision_ind) #= None
        st.session_state.write_db = True

    def remove_decision_callback(decision_ind):
        print("removing decision {decision_ind}")
        st.session_state.db_data[decision_ind]["Decision"] = "none"
        st.session_state.write_db = True

    if st.session_state.write_db:
        print("data base to write is:")
        print(st.session_state.db_data)
        if not os.path.exists(st.session_state.paths["db_path_blocked"]):
            with open(st.session_state.paths["db_path_blocked"], 'w'): pass
            with open(st.session_state.paths["db_path"], 'w') as json_file:
                json.dump(st.session_state.db_data, json_file)
            os.remove(st.session_state.paths["db_path_blocked"])
            st.session_state.write_db = False


    # try retrieving optimizer data

    optimizer_stats = safely_read_db(st.session_state.paths["stats_path"],st.session_state.paths["stats_path_blocked"])

    if not (optimizer_stats is None):
        st.session_state.optimizer_stats = optimizer_stats      


    if st.session_state.optimizer_stats == []:
        st.write("Optimizer Stats could not be retrieved") 
    else:
        with st.expander("**Optimizer Stats**"):
            col1, col2 = st.columns([1, 4])
            with col1:
                for key, value in st.session_state.optimizer_stats[-1]["Numeric Stats"].items():
                    st.write(f"**{key}**: {value}")
            
            with col2:
                #st.write(st.session_state.optimizer_stats[-1]["Numeric Stats"])
                HtmlFile = open(st.session_state.optimizer_stats[-1]["Figure Path"], 'r', encoding='utf-8')
                source_code = HtmlFile.read() 
                st.components.v1.html(source_code,width=st.session_state.optimizer_stats[-1]["config"]["figwidth"], height=st.session_state.optimizer_stats[-1]["config"]["figheight"])

    if os.path.exists(st.session_state.paths["db_path"]) and not os.path.exists(st.session_state.paths["db_path_blocked"]):
        with open(st.session_state.paths["db_path_blocked"], 'w'): pass
        time.sleep(0.1)
        print("results file found")
        with open(st.session_state.paths["db_path"]) as json_file:
            db_data = json.load(json_file)
            st.session_state.db_data = db_data
            print(db_data)
            succesfull_read = True
        os.remove(st.session_state.paths["db_path_blocked"])


    
    if not succesfull_read:
        st.write("There was no database found at " + st.session_state.paths["db_path"] + " ...back end seems to not have been started" )
    else:
    
        # search for data_base entries, that have no associated decision
        decision_ind = 0
        display_Decision = False
        for i in range(len(st.session_state.db_data)):
            if db_data[i]["Decision"] == "none":
                decision_ind = i 
                display_Decision = True
                break
            
        
        if display_Decision and not st.session_state.wait_for_experiment_repetition:
            # Generate two sets of random parameters to compare
            parameter_set_1 = st.session_state.db_data[decision_ind]["Experiment 1"]
            parameter_set_2 = st.session_state.db_data[decision_ind]["Experiment 2"]

            # Display parameter sets with plots to the user
            st.subheader(f"Deciding on Round {decision_ind+1}")
            st.write(f"Choose your preferred parameter set:")

            #load experiment db

            if os.path.exists(st.session_state.paths["experiments_db_path"]) and not os.path.exists(st.session_state.paths["experiments_db_path_blocked"]):

                #with open(db_path_blocked, 'w'): pass
                print("experiments file found")
                with open(st.session_state.paths["experiments_db_path"]) as json_file:
                    experiments_db_data = json.load(json_file)
                    st.session_state.experiments_db_data = experiments_db_data


                print(st.session_state.experiments_db_data)

            for i_experiment in range(len(st.session_state.experiments_db_data["experiments"])):
                if st.session_state.db_data[decision_ind]["Experiment 1"]["Id"] == st.session_state.experiments_db_data["experiments"][i_experiment]["Experiment Id"]: 
                    experiment_1_json = st.session_state.experiments_db_data["experiments"][i_experiment]     
                if st.session_state.db_data[decision_ind]["Experiment 2"]["Id"] == st.session_state.experiments_db_data["experiments"][i_experiment]["Experiment Id"]: 
                    experiment_2_json = st.session_state.experiments_db_data["experiments"][i_experiment]     
                

            figwidth = st.session_state.experiments_db_data["config"]["figwidth"]
            figheight = st.session_state.experiments_db_data["config"]["figheight"]

            print(experiment_1_json)
            print(experiment_2_json)    
               
            col1, col2 = st.columns(2)

            # Plot and display Parameter Set 1
            with col1:
                #st.plotly_chart(plot_parameter(param_set_1), use_container_width=True)
                #st.write(st.session_state.optimization_config["Compare to best"])
                if st.session_state.optimization_config["Compare to best"]:
                    st.write("**Best observed parameters:**")
                else:
                    st.write("**Parameters:**")
                    
                st.write(str(experiment_1_json["Params"]))
                st.write( "Experiment Id: " + str(experiment_1_json["Experiment Id"]))
                if not(experiment_1_json["Figure Path"] == ""):
                    HtmlFile = open(experiment_1_json["Figure Path"], 'r', encoding='utf-8')
                    source_code = HtmlFile.read() 
                    #print(source_code)
                    st.components.v1.html(source_code,width=figwidth ,height=figheight)
                
                st.write("**Numeric results:**")
                for key, value in experiment_1_json["Numeric Results"].items():
                    st.write(f"**{key}**: {value}")
                #st.write(experiment_1_json["Numeric Results"])
                st.button(f"Prefer Set 1", on_click = prefer_set_1_callback, args = (decision_ind, ))
                st.button(f"Repeat Set 1", on_click = repeat_1_callback, args = (decision_ind, ))
                    
            # Plot and display Parameter Set 2
            with col2:
                if st.session_state.optimization_config["Compare to best"]:
                    st.write("**New Parameters:**")
                else:
                    st.write("**Parameters:**")
                st.write(str(experiment_2_json["Params"]))
                st.write( "Experiment Id: " + str(experiment_2_json["Experiment Id"]))
                if len(experiment_2_json["Figure Path"]) > 0:
                    HtmlFile = open(experiment_2_json["Figure Path"], 'r', encoding='utf-8')
                    source_code = HtmlFile.read() 
                    #print(source_code)
                    st.components.v1.html(source_code,width=figwidth ,height=figheight)
                st.write("**Numeric results:**")
                for key, value in experiment_2_json["Numeric Results"].items():
                    st.write(f"**{key}**: {value}")
                st.button(f"Prefer Set 2", on_click = prefer_set_2_callback, args = (decision_ind, ))
                st.button(f"Repeat Set 2", on_click = repeat_2_callback, args = (decision_ind, ))

            col1, col2, col3 = st.columns(3)

            with col1:
                st.button(f"Set 1 crashed", on_click = set_1_crashed_callback, args = (decision_ind, ))
            with col2:
                st.button(f"Both sets crashed", on_click = both_sets_crashed_callback, args = (decision_ind, ))
            with col3:
                st.button(f"Set 2 crashed", on_click = set_2_crashed_callback, args = (decision_ind, ))
        
        with st.expander("**History**"):
        # Step 4: Display the parameter history for transparency, and add "delete" buttons
    
            for i in range(len(st.session_state.db_data)):
                #choice = "Set 1" if st.session_state.preference_history[i] == 0 else "Set 2"
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(json.dumps(st.session_state.db_data[i]))
                with col2:
                    st.button(f"Delete {i}", key=f"delete_{i}", on_click = delete_callback, args = (i, ),use_container_width = True)
                    if st.session_state.db_data[i]["Decision"] == "none":
                        pass
                    else:
                        st.button(f"Remove Decision {i}", key=f"remove_decision_{i}", on_click = remove_decision_callback, args = (i, ),use_container_width = True)
                            #st.experimental_rerun()  # Re-run the app after deleting a choice

        
    # wait for new data
    while True:
        if st.session_state.wait_for_experiment_repetition:
            if not os.path.exists(st.session_state.paths["rerun_right_cmd_path"]) and not os.path.exists(st.session_state.paths["rerun_left_cmd_path"]):
                st.session_state.wait_for_experiment_repetition = False
                st.rerun()
            else:
                time.sleep(2)
                continue        
        
        new_data_read = False
        if os.path.exists(st.session_state.paths["db_path"]) and not os.path.exists(st.session_state.paths["db_path_blocked"]):
            #with open(db_path_blocked, 'w'): pass
            print("results file found")
            with open(st.session_state.paths["db_path"]) as json_file:
                db_data = json.load(json_file)
                if db_data ==  st.session_state.db_data:
                    pass
                else:
                    st.session_state.db_data = db_data
                    new_data_read = True
                    print(db_data)
            #os.remove(db_path_blocked)
        if new_data_read:
            st.rerun()
        time.sleep(2)


if __name__ == "__main__":
    main()
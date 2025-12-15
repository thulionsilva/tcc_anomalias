import pandas as pd
import numpy as np
from opcua import Client, ua
import uuid
from collections import defaultdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import requests
import psycopg2

MAX_ROWS = 10000000  # Maximum number of rows to keep in the DataFrame


#####################################################################################################################

def get_data(
    db_host="localhost",
    db_name="your_database_name", # <--- REPLACE WITH YOUR DATABASE NAME
    db_user="your_username",     # <--- REPLACE WITH YOUR USERNAME
    db_password="your_password", # <--- REPLACE WITH YOUR PASSWORD
    db_port="5432"
):
    """
    Connects to a PostgreSQL database, creates a table if it doesn't exist,
    inserts sample data, queries data from the last month, and plots
    a time series for each variable using Matplotlib.

    Args:
        db_host (str): The PostgreSQL database host.
        db_name (str): The PostgreSQL database name.
        db_user (str): The PostgreSQL username.
        db_password (str): The PostgreSQL password.
        db_port (str): The PostgreSQL port.
        table_name (str): The name of the table to query.
    """
    conn = None
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )
        cursor = conn.cursor()

        print(f"Connected to PostgreSQL database: {db_name} on {db_host}:{db_port}")



        # Execute the query to get data from the last month
        # Using NOW() and INTERVAL '1 month' for precise PostgreSQL date filtering
        cursor.execute(f"""
            SELECT dt, description, value
            FROM variables.iba_data_raw
            WHERE dt >= NOW() - INTERVAL '1 DAY' 
                    and (description ~ '[^H]IC.* - SP Int/Ext' or description ~ '[^H]IC.*PV_IN$' or description ~ '[^H]IC.*SP$' or description ~ '[^H]IC.*MV$')
            ORDER BY dt ASC;
        """)

        # Fetch all results
        results = cursor.fetchall()

        # Print the results
        if results:
            # print("\n--- Last Month's Data ---")
            # for row in results:
            #     print(f"Date: {row[0]}, Variable: {row[1]}, Value: {row[2]}")
            # print("-------------------------")

            # --- Plotting the Data ---
            # Convert results to a Pandas DataFrame
            df = pd.DataFrame(results, columns=['time', 'description', 'value'])
            print("size of df:", df.shape)

            # Ensure 'time' column is datetime type for proper plotting
            # df['time'] = pd.to_datetime(df['time'])

        else:
            print("No data found for the last month to plot.")

    except psycopg2.Error as e:
        print(f"PostgreSQL database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Close the connection
        if conn:
            cursor.close()
            conn.close()
            print("PostgreSQL database connection closed.")
    return df

#####################################################################################################################

def calculate_errors():

    df = get_data(
    db_host="10.55.20.50",
    db_name="postgres", # <--- REPLACE WITH YOUR DATABASE NAME
    db_user="admin",     # <--- REPLACE WITH YOUR USERNAME
    db_password="1234", # <--- REPLACE WITH YOUR PASSWORD
    db_port="5433")


    # Create a copy to avoid SettingWithCopyWarning
    df_mv = df[df['description'].str.endswith('\\MV')].copy()
    df_mv['description'] = df_mv['description'].str.replace(r'\\MV$', '', regex=True)
    # Rename the value column to be specific
    df_mv.rename(columns={'value': 'MV'}, inplace=True)

    df_sp = df[df['description'].str.endswith('\\SP')].copy()
    df_sp['description'] = df_sp['description'].str.replace(r'\\SP$', '', regex=True)
    df_sp.rename(columns={'value': 'SP'}, inplace=True)

    df_pv = df[df['description'].str.endswith('\\PV_IN')].copy()
    # Corrected: Modify the description column of df_pv itself
    df_pv['description'] = df_pv['description'].str.replace(r'\\PV_IN$', '', regex=True)
    df_pv.rename(columns={'value': 'PV_IN'}, inplace=True)

    df_sp_int_ext = df[df['description'].str.endswith(' - SP Int/Ext')].copy()
    # Corrected: Modify the description column of df_sp_int_ext itself
    df_sp_int_ext['description'] = df_sp_int_ext['description'].str.replace(r' - SP Int/Ext$', '', regex=True)
    # Corrected: Create the 'is_manual' column based on its own 'value' column
    df_sp_int_ext['is_manual'] = (df_sp_int_ext['value'].astype(int) & 128) > 0
    df_sp_int_ext.rename(columns={'value': 'value_sp_int_ext'}, inplace=True)


    # Start with your "left" dataframe, for instance df_sp
    merged_df = df_sp

    # List of the other dataframes to join
    dfs_to_join = [df_mv, df_pv, df_sp_int_ext]

    # Loop through the list and merge each one
    for df_to_join in dfs_to_join:
        merged_df = pd.merge(
            merged_df,
            df_to_join,
            on=['time', 'description'],
            how='left'
        )

    # Display the first few rows of the final merged dataframe
    #print(merged_df.head())

    # Define the conditions for the calculation
    # Condition 1: is_manual is True
    cond1 = merged_df['is_manual'] == True

    # Condition 2: SP is zero (to prevent division errors)
    cond2 = merged_df['SP'] == 0

    # Calculate the error using nested np.where
    # If cond1 is true -> error is 0
    # Else, if cond2 is true -> error is NaN (or 0 if you prefer)
    # Else -> perform the calculation
    merged_df['error'] = np.where(cond1,
                                0,
                                np.where(cond2,
                                        np.nan, # Assign NaN for division by zero
                                        (merged_df['SP'] - merged_df['PV_IN']) / merged_df['SP'])
                                )

    # Display the dataframe with the new column
    #print(merged_df[['description', 'SP', 'PV_IN', 'is_manual', 'error']].head())
    last_values_df = merged_df.sort_values('time').drop_duplicates(subset=['description'], keep='last')

    return merged_df, last_values_df

#####################################################################################################################

def get_tag_rules(
    db_host="localhost",
    db_name="your_database_name", # <--- REPLACE WITH YOUR DATABASE NAME
    db_user="your_username",     # <--- REPLACE WITH YOUR USERNAME
    db_password="your_password", # <--- REPLACE WITH YOUR PASSWORD
    db_port="5432"
):
    """
    Connects to a PostgreSQL database, creates a table if it doesn't exist,
    inserts sample data, queries data from the last month, and plots
    a time series for each variable using Matplotlib.

    Args:
        db_host (str): The PostgreSQL database host.
        db_name (str): The PostgreSQL database name.
        db_user (str): The PostgreSQL username.
        db_password (str): The PostgreSQL password.
        db_port (str): The PostgreSQL port.
        table_name (str): The name of the table to query.
    """
    conn = None
    results = []
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )
        cursor = conn.cursor()

        print(f"Connected to PostgreSQL database: {db_name} on {db_host}:{db_port}")



        # Execute the query to get data from the last month
        # Using NOW() and INTERVAL '1 month' for precise PostgreSQL date filtering
        cursor.execute(f"""
            SELECT tag, rule_type, threshold_value1, threshold_value2, rolling_window
            FROM plant_monitor.tag_rules
        """)

        # Fetch all results
        results = cursor.fetchall()

    except psycopg2.Error as e:
        print(f"PostgreSQL database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Close the connection
        if conn:
            cursor.close()
            conn.close()
            print("PostgreSQL database connection closed.")
    return pd.DataFrame(results, columns=['tag', 'rule_type', 'threshold_value1', 'threshold_value2', 'rolling_window'])

########################################################################################################
def discover_grouped_variables(folder_node, suffixes=("\\PV_IN", "\\SP", "\\MV", " - SP Int/Ext")):
    """
    Browse recursively and collect variables whose name ends with one of the suffixes.
    Returns a dict:
    {
        "TagName": {
            "PV_IN": <Node>,
            "SP": <Node>,
            "MV": <Node>
        }
    }
    """
    grouped = defaultdict(dict)

    for child in folder_node.get_children():
        node_class = child.get_node_class()

        if node_class == ua.NodeClass.Variable:
            name = child.get_browse_name().Name.split(" ")[1]
            for suffix in suffixes:
                if name.endswith(suffix):
                    tag_name = name.replace(suffix, "")
                    clean_suffix = suffix.strip("\\").strip(" - ")  # "\PV_IN" -> "PV_IN"
                    grouped[tag_name][clean_suffix] = child

        elif node_class == ua.NodeClass.Object:
            sub_group = discover_grouped_variables(child, suffixes)
            for k, v in sub_group.items():
                grouped[k].update(v)

    return dict(grouped)

###########################################################################################
def get_new_rows(variables):
    """
    Calculate (PV_IN - SP) / SP for each tag and collect PV_IN, SP, MV.
    Returns a DataFrame with Tag, PV_IN, SP, MV, Error.
    """
    data = []
    for tag, nodes in variables.items():
        try:
            pv = nodes["PV_IN"].get_value() if "PV_IN" in nodes else None
            sp = nodes["SP"].get_value() if "SP" in nodes else None
            mv = nodes["MV"].get_value() if "MV" in nodes else None
            is_manual = (nodes["SP Int/Ext"].get_value() & 128) > 0  if "SP Int/Ext" in nodes else False
        except Exception:
            pv, sp, mv, is_manual = None, None, None, None

        if sp not in (None, 0):
            error = (pv - sp) / sp if pv is not None else None
        else:
            error = None

        data.append({
            "time": (pd.Timestamp.now()).tz_localize('UTC'),
            "description": tag,
            "PV_IN": pv,
            "SP": sp,
            "MV": mv,
            "is_manual": is_manual,
            "error": error
        })

    return pd.DataFrame(data)

######################################################################
def update_dataframe(df, new_rows):
    """Append new row and keep DataFrame at MAX_ROWS size"""
    min_timestamp = new_rows.iloc[0,0] - pd.Timedelta(60, 'days')
    df = pd.concat([df, new_rows], ignore_index=True)
    df2 = df[df['time'] > min_timestamp]
    # if len(df) > MAX_ROWS:
    #    df = df.iloc[-MAX_ROWS:]  # keep only latest MAX_ROWS
    return df2
###########################################################################################
def filter_setpoint_changes(group, tag_rules_df):
    """
    Filters out rows where the setpoint (SP) has changed for each description.
    Returns a DataFrame with only rows where SP has not changed.
    """
    # now = time.time()
    now = pd.Timestamp.now()
    # print(f"start time for filter_setpoint_changes: {now}")
    group_name = group['description'].iloc[0]
    rule_row = tag_rules_df[tag_rules_df['tag'] == group_name]

    window_mv_std = rule_row[rule_row['rule_type'] == 'mv_std']['rolling_window'].values
    # print(f"Applying filter for group: {group_name} with window_mv_std: {window_mv_std}")
    window_error_cv = rule_row[rule_row['rule_type'] == 'error_cv']['rolling_window'].values
    window_mv_slope = rule_row[rule_row['rule_type'] == 'mv_slope']['rolling_window'].values
    if len(window_mv_std) > 1:
        window_mv_std = int(window_mv_std[0])
    else:
        window_mv_std = 360

    if len(window_error_cv) > 1:
        window_error_cv = int(window_error_cv[0])
    else:
        window_error_cv = 360

    group = group.sort_values('time')
    group.set_index('time', inplace=True)
    group['mv_std'] = group['MV'].rolling(window=360, min_periods=1).std()
    group['trend_slope'] = group['MV'].rolling(window=360, min_periods=1).apply(rolling_slope, raw=False)
    group['>trend'] = abs(group['trend_slope']) >= abs(group['trend_slope'].max())

    group['error_trend_slope'] = group['error'].rolling(window=360, min_periods=1).apply(rolling_slope, raw=False)
    group['error_std'] = group['error'].rolling(window=360, min_periods=1).std()
    group['error_mean'] = group['error'].rolling(window=360, min_periods=1).mean()
    group['error_cv'] = group['error_std'] / group['error_mean'].replace(0, np.nan)
    group['error_lsc'] = group['error_mean'] + 4 * group['error_std']
    group['error_usc'] = group['error_mean'] - 4 * group['error_std']
    group['error_out_of_control'] = ((group['error'] > group['error_lsc']) | (group['error'] < group['error_usc'])) & (abs(group['error']) > 0.05)

    group['>error_trend'] = abs(group['error_trend_slope']) >= abs(group['error_trend_slope'].max())
    group['sp_change'] = group['SP'].rolling(window=360, min_periods=1).std().fillna(0).eq(0)
    # print(group.shape, group_name)
    # print(group['sp_change'])
    stable_sp_rows = group[group['sp_change'] & (group['is_manual'] == False)]
    return stable_sp_rows
###########################################################################################
def apply_filter_setpoint_changes(df, tag_rules_df):
    """
    Apply the filter_setpoint_changes function to each group of description in the DataFrame.
    Returns a filtered DataFrame.
    """
    filtered_groups = []
    for description, group in df.groupby('description'):
        filtered_group = filter_setpoint_changes(group, tag_rules_df)
        filtered_groups.append(filtered_group)
    filtered_df = pd.concat(filtered_groups, ignore_index=False)
    filtered_df = filtered_df.reset_index()
    return filtered_df

###########################################################################################

def send_state_change_email(malha, which_message):
    """
    Sends an email notification when the state of a malha changes.
    Args:
        malha (str): The name of the malha (loop/variable).
        old_state (bool): Previous state (True = stuck, False = normal).
        new_state (bool): New state (True = stuck, False = normal).
        recipient (str): Recipient email address.
        smtp_server (str): SMTP server address.
        smtp_port (int): SMTP server port.
        smtp_user (str): SMTP username.
        smtp_password (str): SMTP password.
    """
    recipient = ['Recipiente@poli.br']
    smtp_server = 'smtp.office365.com'  # Example SMTP server for Office 365
    smtp_port = 587  # Common SMTP port for TLS
    smtp_user = 'Email@dominio.com'  # <--- REPLACE WITH YOUR SMTP USERNAME
    smtp_password  = 'ChlorumSenha'  # <--- REPLACE WITH YOUR SMTP PASSWORD
    subject = f"Alerta Erro de Controle - {malha}"


    if which_message == 1:
        message = f"Erro maior que 5%.\n\nThis email is sent automatically by the Chlorum system."
        body = (
            f"Alto Erro Percentual - '{malha}'.\n\n"
            f"{message}"
        )
    elif which_message == 2:
        message = f"Erro maior que 10%.\n\nThis email is sent automatically by the Chlorum system."
        body = (
            f"Alto Erro Percentual - '{malha}'.\n\n"
            f"{message}"
        )
    elif which_message == 3:
        message = f"Erro fora de controle.\n\nThis email is sent automatically by the Chlorum system."
        body = (
            f"Erro Fora de Controle - '{malha}'.\n\n"
            f"{message}"
            )
    elif which_message == 4:
        message = f"Tendencia MV.\n\nThis email is sent automatically by the Chlorum system"
        body = (
            f"Tendencia MV - '{malha}'.\n\n"
            f"{message}"
            )
    elif which_message == 5:
        message = f"Aplitude do Erro%.\n\nThis email is sent automatically by the Chlorum system."
        body = (
            f"Alto Amplitude de Erro - '{malha}'.\n\n"
            f"{message}"
        )
    elif which_message == 6:
        message = f"Atuador no limite.\n\nThis email is sent automatically by the Chlorum system."
        body = (
            f"Altuador no limite - '{malha}'.\n\n"
            f"{message}"
        )


    if len(recipient) > 1:
        joined_emails = ', '.join(recipient)
    else:
        joined_emails = recipient[0]

    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = joined_emails
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, recipient, msg.as_string())
        print(f"Email sent to {recipient} for malha {malha} state change.")
    except Exception as e:
        print(f"Failed to send email: {e}")

###########################################################################################

def send_state_ntfy(malha, which_message):


    if which_message == 1:
        message = f"Erro maior que 5%.\n\nThis email is sent automatically by the Chlorum system."
        Title = f"Alerta Erro de Controle - {malha}"
    elif which_message == 2:
        message = f"Erro maior que 10%.\n\nThis email is sent automatically by the Chlorum system."
        Title = f"Alerta Erro de Controle - {malha}"
    elif which_message == 3:
        message = f"Erro fora de controle.\n\nThis email is sent automatically by the Chlorum system."
        Title = f"Alerta Erro de Controle - {malha}"
    elif which_message == 4:
        message = f"Tendencia MV.\n\nThis email is sent automatically by the Chlorum system"
        Title = f"Tendencia MV - {malha}"

    try:
        requests.post("https://ntfy.sh/chlorum_solutions_alert",
    data=message,
    headers={
        "Title": Title,
        "Priority": "urgent",
        "Tags": "warning,skull"
    })

    except Exception as e:
        print(f"Failed to send alert")
###########################################################################################################################

def rolling_slope(window_df):
    # window_df will be a DataFrame subset with 'timestamp' and 'value' columns
    # You must convert the timestamps to a numeric format (e.g., seconds since epoch)
    #X = window_df['time'].apply(lambda t: t.timestamp())  # Example conversion
    X = window_df.index.astype(np.int64) // 10**9  # Convert index (datetime) to seconds since epoch
    Y = window_df.values

    # Perform linear regression (e.g., using NumPy's polyfit)
    if len(X) < 5:
        return 0
    slope, intercept = np.polyfit(X, Y, 1) # 1 is the degree of the polynomial (linear)
    return slope

###########################################################################################################################
def error_out_of_control(description, last_values_df):

    if last_values_df["error_out_of_control"]:
        print(f"Malha:{description}, Error: {last_values_df['error']}, Modo: {last_values_df['value_sp_int_ext']}, SP: {last_values_df['SP']}, MV: {last_values_df['MV']}, PV: {last_values_df['PV_IN']}")
        send_state_change_email(description, 3)
        # send_state_ntfy(description, 3)


def detect_error_5(description, last_values_df, flagged):
    try:
        if not flagged.loc[description, 'flagged'] and abs(last_values_df["error"]) > 0.05:
            print(f"Malha {description} Error: {last_values_df['error']:.2%}, MV: {last_values_df['MV']:.2f}. Sending alert.")
            send_state_change_email(description, 1)
            # send_state_ntfy(description, 1)

            flagged.loc[description, 'flagged'] = True
            flagged.to_csv('flagged.csv', index=True)

        if flagged.loc[description, 'flagged'] and abs(last_values_df["error"]) < 0.05:
            print(f"Malha {description} voltou a funcionar corretamente.")
            flagged.loc[description, 'flagged'] = False
            flagged.to_csv('flagged.csv', index=True)
    except KeyError:
        print(f"New malha detected: {description}. Adding to tracking list.")
        flagged.loc[description, 'flagged'] = False

###########################################################################################################################
def detect_error_10(description, last_values_df, flagged):
    try:
        if not flagged.loc[description, 'flagged'] and abs(last_values_df["error"]) > 0.1:
            print(f"Malha:{description}, Error: {last_values_df['error']}, Modo: {last_values_df['value_sp_int_ext']}, SP: {last_values_df['SP']}, MV: {last_values_df['MV']}, PV: {last_values_df['PV_IN']}")
            # print(f"Malha {description} Error: {last_values_df['error']:.2%}, MV: {last_values_df['MV']:.2f}. Sending alert.")
            send_state_change_email(description, 2)
            # send_state_ntfy(description, 2)

            flagged.loc[description, 'flagged'] = True
            flagged.to_csv('flagged.csv', index=True)

        if flagged.loc[description, 'flagged'] and abs(last_values_df["error"]) < 0.1:
            print(f"Malha {description} voltou a funcionar corretamente.")
            flagged.loc[description, 'flagged'] = False
            flagged.to_csv('flagged.csv', index=True)
    except KeyError:
        print(f"New malha detected: {description}. Adding to tracking list.")
        flagged.loc[description, 'flagged'] = False

###########################################################################################################################

def detect_slope(description, last_values_df, flagged_mv):
    try:
        if not flagged_mv.loc[description, 'flagged'] and last_values_df[">trend"] and last_values_df["error"]:

            # send_state_ntfy(description, 4)

            flagged_mv.loc[description, 'flagged'] = True
            flagged_mv.to_csv('flagged_mv.csv', index=True)

        if flagged_mv.loc[description, 'flagged'] and not last_values_df[">trend"]:
            flagged_mv.loc[description, 'flagged'] = False
            flagged_mv.to_csv('flagged_mv.csv', index=True)
    except KeyError:
        print(f"New malha detected: {description}. Adding to tracking list.")
        flagged_mv.loc[description, 'flagged'] = False

###########################################################################################################################

def detect_stuck_valve(description, last_values_df, flagged_stuck):
    try:
        if not flagged_stuck.loc[description, 'flagged'] and (last_values_df["MV"] > 99 or last_values_df["MV"] < 1) and last_values_df["is_manual"]==False:
            print(f"Malha:{description}, Error: {last_values_df['error']}, Modo: {last_values_df['value_sp_int_ext']}, SP: {last_values_df['SP']}, MV: {last_values_df['MV']}, PV: {last_values_df['PV_IN']}")

            # send_state_ntfy(description, 4)
            send_state_change_email(description, 6)

            flagged_stuck.loc[description, 'flagged'] = True
            flagged_stuck.to_csv('flagged_stuck.csv', index=True)

        if flagged_stuck.loc[description, 'flagged'] and last_values_df["MV"] < 99 and last_values_df["MV"] > 1:
            flagged_stuck.loc[description, 'flagged'] = False
            flagged_stuck.to_csv('flagged_stuck.csv', index=True)
    except KeyError:
        print(f"New malha detected: {description}. Adding to tracking list.")
        flagged_stuck.loc[description, 'flagged'] = False

###########################################################################################################################
def detect_error_slope(description, last_values_df, flagged_mv):
    try:
        if not flagged_mv.loc[description, 'flagged'] and last_values_df[">error_trend"] and last_values_df["error"]:

            send_state_ntfy(description, 3)

            flagged_mv.loc[description, 'flagged'] = True
            flagged_mv.to_csv('flagged.csv', index=True)

        if flagged_mv.loc[description, 'flagged'] and not last_values_df[">error_trend"]:
            flagged_mv.loc[description, 'flagged'] = False
            flagged_mv.to_csv('flagged.csv', index=True)
    except KeyError:
        print(f"New malha detected: {description}. Adding to tracking list.")
        flagged_mv.loc[description, 'flagged'] = False

###########################################################################################################################
def detect_error(description, last_values_df, flagged):
    error_out_of_control(description, last_values_df)
    #detect_error_5(description, last_values_df, flagged)
    detect_error_10(description, last_values_df, flagged)
###########################################################################################################################

#flagged = df_mv.drop(columns='time').groupby('description').std().eq(0).rename(columns={'value': 'flagged'})
#flagged.set_index('description', inplace=True)
#flagged.to_csv('flagged.csv', index=True)

#Store sent emails to avoid spamming
#Only send again if error goes back to normal and then fails again

skip_email = ['QIC10720', 'LIC076015', 'PDIC11152A', 'PDIC11152B', 'PIC11151B', 'PIC11151A', 'PIC11151B', 'PIC02121A', 'PIC02121B', 'QIC08101', 'LIC05111']  # Add malhas that should not send emails

url = "opc.tcp://10.55.20.88:4880"
folder_path = ["0:Objects", "2:ibaPDA", "3:Modules"]
client = Client(url)

client.connect()
print(f"Connected to {url}")
folder_node = client.get_root_node().get_child(folder_path)
variables = discover_grouped_variables(folder_node)
print("finnished scanning for tags")


# df = get_new_rows(variables)
df, df_last = calculate_errors()
print("First new rows")
flagged = df_last[['description']]
flagged['flagged'] = False
#flagged = pd.read_csv('flagged.csv')
flagged.set_index('description', inplace=True)

flagged_mv = df_last[['description']]
flagged_mv['flagged'] = False
#flagged_mv = pd.read_csv('flagged.csv')
flagged_mv.set_index('description', inplace=True)


flagged_stuck = df_last[['description']]
flagged_stuck['flagged'] = False
# flagged_stuck = pd.read_csv('flagged.csv')
flagged_stuck.set_index('description', inplace=True)

tag_rules_df = get_tag_rules(
    db_host="10.55.20.50",
    db_name="postgres", # <--- REPLACE WITH YOUR DATABASE NAME
    db_user="admin",     # <--- REPLACE WITH YOUR USERNAME
    db_password="1234", # <--- REPLACE WITH YOUR PASSWORD
    db_port="5433")
try:
    while True:
        now = time.time()
        print(f"new loop: {pd.Timestamp.now()}")
        new_rows = get_new_rows(variables)
        print(f"new rows shape {new_rows.shape}")
        print(f"old df shape {df.shape}")
        df = update_dataframe(df, new_rows)
        print(f"df shape after concat {df.shape}")
        df = apply_filter_setpoint_changes(df, tag_rules_df)
        print(f"df shape after filter {df.shape}")

        for description, group in df.groupby('description'):
            #print(group)
            last_values_df = group.iloc[-1]
            if description in skip_email:# or row.is_manual:
                # print(f"Skipping malha {description} for email alert.")
                continue  # Skip malhas that should not send emails
            
            detect_error(description, last_values_df, flagged)
            detect_slope(description, last_values_df, flagged_mv)
            detect_stuck_valve(description, last_values_df, flagged_stuck)
        
        print(f"A loop took {time.time() - now} seconds")
        time.sleep(10)  # Espera 10 segundos antes de verificar novamente
    
except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    flagged.to_csv('flagged.csv', index=True)       
    flagged_mv.to_csv('flagged_mv.csv', index=True)       
    flagged_stuck.to_csv('flagged_stuck.csv', index=True)       
    client.disconnect()
    print("Disconnected")


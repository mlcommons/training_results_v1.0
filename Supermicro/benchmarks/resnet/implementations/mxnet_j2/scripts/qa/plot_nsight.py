################################################################################
##
##  plot_nvprof
##
##                 - Automatically generate a python script
##                 - Script will open files for reading and writing
##                 - And will have examples of basic operations like formatted
##                 - printing
##
##  asettle
##  Fri Nov 3 2017
##
##  Kai Ma
##  Mon Aug 7 2019
################################################################################
import getopt
import sys
import sqlite3
import re
import os
import pandas as pd
import subprocess
import json
import time
from yaml import load, dump, Loader

################################################################################
## Algorithm to Link GPU events to Layer names
##
##  For each GPU event in CONCURRENT_KERNELS
##  Use the correlation ID to map the GPU event to the runtime cuda event (function call)
##  Now for this Runtime event - record the start time and end time
##  Then Go to the markers - find all markers whose start times are > runtime event
##  start and whose end time is > runtime event end
##   There should be 1 Marker that meets this criteria
################################################################################


################################################################################
## Helper function definitions
################################################################################
def uniq_layer_name(layer_name):
    # return ''.join(e for e in lstr if e not in ["*", "[", "]", "(", ")", " ", ",", ":"])
    # Strip off " [profile *]" from the layer name
    # (layer name with/without [profile *] should be the same from DLSim perspective)
    layer_name = re.sub(r"\s\[profile\s\d+\]", '', layer_name)
    return layer_name


def convert_uniq_lname(lnamestr):
    """remove all non-number non-alphabet chars from a string."""

    return ''.join(e for e in lnamestr if e not in ["*", "[", "]", "(", ")", " "])


def clean_conditional_autograd_lname(l_name, l_type):
    """ clean autograd layer names for conditional ops."""
    l_name = re.sub("/cond_grad/", "/cond/", l_name)
    l_name = re.sub("/StatelessIf/", "/", l_name)
    for cond_str in ["/cond/else", "cond/then"]:
        l_name = re.sub(r"{}/_[0-9]+/gradients".format(cond_str), cond_str, l_name)
    # Many layers of this type are wrongly classified as Identity instead of actual type
    # (mul in this case)
    # Adam/gradients/gradients/.../IdentityN_grad/mul_2
    if "/IdentityN_grad/" in l_name and l_type != "Identity":
        l_name = re.sub("/IdentityN_grad/", "/{}_grad/".format(l_type), l_name)

    return l_name
def decode_object_id(obj_kind, obj_byte_array):
    """
    Read in the object byte array.
    The format is ProcID:ThreadID.
    ProcID is 32 bits and threadID is 64 Bits.
    The bytes in byte array are in reverse order.
    """
    pid = 0
    th_id = 0
    if obj_kind != 2:
        print("Error - unexpected obk_kind val -> {}, expecting 2".format(obj_kind))
        sys.exit(1)

    reverse_proc_id = obj_byte_array[:3]  ## Proc ID is 32 bits (4 bytes)
    reverse_th_id = obj_byte_array[4:]  ## Thread ID is 64 bits - just take all the remaining bytes

    pid = int.from_bytes(reverse_proc_id, byteorder='little')
    th_id = int.from_bytes(reverse_th_id, byteorder='little')

    #print ("ProcId -> {} Thread ID -> {}".format(pid, th_id))

    return [pid, th_id]


def get_tbl_hdrs(cursor=None, display=True):
    """
    Return the column headers from the sql table
    """
    tbl_hdr = {}  ## Hash table to map col header to index
    for idx, col in enumerate(cursor.description):
        if (display):
            print("Col Header: {0} index {1}".format(col[0], idx))
        tbl_hdr[col[0]] = idx
    if (display):
        ## Prtint the header in 1 row
        for idx, col in enumerate(cursor.description):
            print("{0} ".format(col[0]))
        print("")
    return tbl_hdr


def time_stamp_to_duration(ts_measured=None, ts_base=None, scale_factor=1):
    """Takes two time stamps and an optional scale factor and converts to a duration of time."""
    if scale_factor == 0:
        print("Error divide by 0 - exiting")
        sys.exit(1)

    if ts_measured is None or ts_base is None:
        print("Error bad arguments: either ts_measured or ts_base is Null")
        sys.exit(1)

    time = (ts_measured - ts_base) / scale_factor
    return time


def get_unique_tags_from_frame(field_name, pd_frame):
    """
    Return a list of unique tags from a specific field in a pandas frame.
    Eg- for a col named 'Layers' - return a list of the unique layer names.
    """
    unique_tags = []
    tag_name_list = pd_frame[field_name].tolist()

    for tag in tag_name_list:
        if tag not in unique_tags:
            unique_tags.append(tag)
    return unique_tags

def detect_fw_type(net_name):
    fw_type = None
    if re.match(r"\w+[/]\S+", net_name):
        fw_type = "FW_TENSOR_FLOW"
    elif re.match(r"N\d+torch", net_name):
        fw_type = "FW_PYTORCH"
    elif re.match(r".+[(].+", net_name):
        fw_type = "FW_TENSORRT"
    return fw_type


################################################################################
## Reserved functions
################################################################################
def get_tbl_event_by_corr_id(corr_id=None, pd_frame=None):
    if corr_id is None or pd_frame is None:
        print("Error get_runtime_event_by_corr_id: missing argument - exiting.\n")
        sys.exit(1)
    ## use panda frame instead of sql query
    query_string = "correlationId == {}".format(corr_id)
    tmp_frame = pd_frame.query(query_string)
    if tmp_frame.empty:
        raise Exception("Query {} failed for lookup in RUNTIME and DRIVER table ".format(query_string))

    start = tmp_frame['start'].iat[0]
    end = tmp_frame['end'].iat[0]
    thread_id = tmp_frame['threadId'].iat[0] % 0x1000000

    return [start, end, thread_id]


def tbl_name_lookup_by_id(cur=None, name_id=None):
    if name_id is None:
        print("Error name_lookup_by_id - no name specified - exiting...")
        sys.exit(1)

    if cur is None:
        print("Error process_runtime_tbl: No cursor specified - exiting.\n")
        sys.exit(1)

    query_string = "select value from StringIds where _id_={0}".format(name_id)

    return


def run_io_examples():
    """Examples of print usage"""
    ## Open a file for reading and write to it
    file_des = open("tmp.txt", "w")

    ## unformatted print
    print("## Example python file writes", file=file_des)
    print("## Unformatted prints", file=file_des)
    print(65, "F", sep="...", file=file_des)
    print((65 - 32) * 5 / 9, "C", sep=" ?", file=file_des)


class NsysParser:
    """
    This calss is for exporting .sqlite file to .xlsx with profiling info.
    Integrate existing functions and wrap to a class, with better readability and extensibility.
    """

    Supported_Fw = ["FW_TENSOR_FLOW", "FW_TENSORRT", "FW_PYTORCH", "FW_MXNET", "FW_CAFFE2"]
    MAX_EXCEL_SHEET_LEN = 31
    MAX_INT32 = 1 << 32

    def __init__(self):

        self.db_file_list = []  ## Input file seql DB
        self.pivot_tbl = None  ## Output pivot table
        self.excel_file_name = None
        self.string_hash = {}  ## Hash table - maps string ID to name
        self.kernel_hash = {}  ## Hash table - stores demangled kernel names
        self.time_base = -1  ## Starting time stamp of experiment
        self.Debug = False  ## True for print debugging
        self.ComputeAverage = True  ## False for disabling the average of per layer times
        self.phase          = 'Fprop' ## Start off in Fprop - then transition the phases based on
        # Used to prune away entries
        self.prune_enable = False
        self.prune_marker = ""
        self.print_all_tbls = False
        self.HeartBeat = 0
        self.FwType = None
        self.graph_input_file = None
        self.graph_info_map = {}
        self.xla_kernel_file = ""
        self.xla_kernel_map  = {}
        self.xla_enable      = False
        self.weight_to_fprop_file = None
        self.weight_to_fprop_map = {} ## Mapping of trainable var to fprop op
        self.export_trt_json = False
        self.json_file_name = None
        ## PyTorch
        self.min_seq_id = -1
        # MXNET
        self.mxnet_phase = 'fprop'
        self.mxnet_print_interval = 1000
        self.mxnet_fwd_layer_name_dict = {}
        self.mxnet_bwd_layer_name_dict = {}
        self.prev_layer_name = None
        self.mxnet_nvtx_events = 'mxnet_nvtx_events.csv'
        self.saved_nvtx_strings = []


    def parse_cmd_line(self):
        """Uses getopt to parse cmd line options from user."""

        options = 'h:o:i'  ## Help message - string of possible 1 char options, ':' after option means it takes an arg
        long_options = [
            'in_files=', 'out_file=', 'debug', 'help', 'show_tables', 'heartbeat=', 'prune=', 'no_average', 'graph=',
            'framework=', 'export-trt-json', 'weight_to_fprop=', 'xla_map='
        ]  ## List of long form options

        ## Exception handling
        try:
            opts, extra_args = getopt.gnu_getopt(sys.argv[1:], options, long_options)
        except getopt.GetoptError as err:
            print("Exception caught :  {0}".format(err))  ## Didn't specify type of err in format specifier
            sys.exit(1)

        ## Walk list of cmd line options - opts is a pair<string,string>
        for opt, arg in opts:
            if (opt == "-i" or opt == "--in_files"):
                self.db_file_list = re.split(',', arg)
                print("Reading in_file {0}".format(arg))
            elif (opt == "--export-trt-json"):
                self.export_trt_json = True
            elif (opt == "-o" or opt == "--out_file"):
                print("Writing out file {0:s}".format(arg))
                self.pivot_tbl = arg
                self.excel_file_name = re.sub(r'.txt', r'.xlsx', self.pivot_tbl)
                self.json_file_name = re.sub(r'.txt', r'.json', self.pivot_tbl)
                self.result_dir = os.path.dirname(self.excel_file_name)
            elif (opt == "-h" or opt == "--help"):
                print("Usage: plot_nsight [-h] --in_files nvp_sqlite_file,nvp_file1,nvp_file2 -out_file output_file_name [--show_tables] [--debug] [--framework] [--prune]")
                sys.exit(0)
            elif (opt == "-d" or opt == "--debug"):
                print("Enabling Debug print messages")
                self.Debug = True
            elif (opt == "-s" or opt == "--show_tables"):
                self.Debug = True
                self.print_all_tbls = True
            elif (opt == "--graph"):
                self.graph_input_file = arg
            elif (opt == "--weight_to_fprop"):
                self.weight_to_fprop_file = arg            
            elif (opt == "--framework"):
                self.FwType = arg
                if self.FwType not in NsysParser.Supported_Fw:
                    raise Exception("Framework option {} not supported, must be one of {}", self.FwType,
                                    NsysParser.Supported_Fw)
            elif (opt == "-b" or opt == "--heartbeat"):
                self.HeartBeat = int(arg)
            elif (opt == "-a" or opt == "--no_average"):
                self.ComputeAverage = False
            elif (opt == '--prune'):
                self.prune_enable = True
                self.prune_marker = arg
            elif (opt == '--xla_map'):
                self.xla_kernel_file = arg
                self.xla_enable = True

    def open_ouput_file(self):
        """Check to see if output file specified on cmd line, else use stdout."""

        ## Open the output file (pivot_table)
        if (self.pivot_tbl is None):
            file_des = sys.stdout
        else:
            file_des = open(self.pivot_tbl, "w")
        return file_des

    def reset_global_vars(self):
        self.time_base = -1
        self.string_hash = {}

    def read_db_file(self, db_name=None, output_fd=None):
        """Read in the DB file and extract relevant tables."""
        if db_name is None or output_fd is None:
            print("Error read_db_file: No db file specified - exiting. ")
            sys.exit(1)
        if not os.path.isfile(db_name):
            print("Error read_db_file: file {} not found".format(db_name))
            sys.exit(1)

        print("Reading DB file {0}".format(db_name))
        connection = sqlite3.connect(db_name)
        cur = connection.cursor()
        cur.execute("select name from sqlite_master where type='table'")
        #dump_cur(cur)
        all_tbls = self.get_tbl_names(cur)
        print("All tables {}".format(all_tbls))
        remaining_tbls = []

        ## Read in StringIds and DRIVER first to extract global info used by other table processing
        for tbl in all_tbls:
            update_list = 1
            if not self.print_all_tbls:
                #for tbl_type in ['DRIVER', 'StringIds', 'RUNTIME', 'MARKER$']:
                #pattern = re.compile(tbl_type)
                #pattern = re.compile(r"(DRIVER|StringIds|RUNTIME|MARKER$)")
                pattern = re.compile(r"(DRIVER|StringIds)")
                res = re.search(pattern, tbl)
                if res is not None:
                    tbl_type = res.group(1)
                    print("Processing table {}".format(tbl_type))
                    self.process_tbl(tbl, cur, tbl_type)
                    update_list = 0
            if update_list:
                remaining_tbls.append(tbl)

        # Walk the remaining list of tables
        if (self.Debug or self.print_all_tbls):
            for tbl in remaining_tbls:
                print("Tbl {0:s}".format(tbl))
                #process_runtime_tbl(tbl, cur)
                tbl_str = re.sub(r".*_KIND_", "", tbl)
                print("tbl str {0} from table {1}".format(tbl_str, tbl))
                print("Processing table {}".format(tbl_str))
                self.process_tbl(tbl, cur, tbl_str)

        if (self.print_all_tbls):
            print("Option --show_tables set - exiting after printing tables")
            sys.exit(0)

        ## Layer names (CPU Runtime) to the kernels (GPU) that they launch
        panda_frame = self.link_kernel_to_dl_layer(cur, all_tbls, db_name, output_fd)
        if self.prune_enable:
            panda_frame = self.prune(panda_frame)

        ## Emit yml file w/ nvtx marker strings
        if self.FwType == 'FW_MXNET':
            mxnet_nvtx_frame = self.get_marker_pandas_tbl_frame('NVTX_EVENTS', cur)
            mxnet_nvtx_frame.to_csv(self.mxnet_nvtx_events, sep='|')

        connection.close()

        ## Clear globals that are set up on each pass of the db file
        self.reset_global_vars()

        return panda_frame

    def prune(self, df):
        """Use prune_marker to filter out entries from a dataframe."""
        cret1 = df['LayerName'] == self.prune_marker
        cret2 = df['Phase'] == 'wgrad'
        df1 = df.index[cret1 & (cret2)]
        #print(df)
        #print(df1)
        return df[df1[2]:]

    def dump_rows(self, cursor=None, tbl_hdr=None, tbl_type=None):
        """Walk all the rows in the table and prinstr."""
        if cursor is None:
            print("Error dump_rows: No cursor specified - exiting.")
            sys.exit(1)

        if tbl_hdr is None:
            print("Error dump_rows: No col headers specified - exiting.")
            sys.exit(1)

        if tbl_type is None:
            print("Error dump_rows: No table type name specified- exiting.")
            sys.exit(1)

        ## Check the tbl_type - call the tbl specific dump function
        if (tbl_type == 'RUNTIME') or (tbl_type == 'DRIVER'):
            self.dump_rows_runtime_driver(cursor, tbl_hdr, tbl_type)
        elif tbl_type == 'NAME':
            self.dump_rows_name(cursor, tbl_hdr, tbl_type)
        elif tbl_type == 'StringIds':
            self.dump_rows_strings(cursor, tbl_hdr, tbl_type)
        elif tbl_type == 'MARKER':
            self.dump_rows_marker(cursor, tbl_hdr, tbl_type)
        elif tbl_type == 'KERNEL':
            self.dump_rows_conc_kernel(cursor, tbl_hdr, tbl_type)
        else:
            self.dump_rows_default(cursor, tbl_hdr, tbl_type)

        return

    def dump_rows_default(self, cur=None, hdr=None, tbl_type=None):
        """Dump the contents of the sql cursor for tbl type NAME."""
        if cur is None:
            print("Error dump_rows_default: No cursor specified - exiting.")
            sys.exit(1)

        if hdr is None:
            print("Error dump_rows_default: No col headers specified - exiting.")
            sys.exit(1)

        if tbl_type is None:
            print("Error dump_rows_default: No table type name specified - exiting.")
            sys.exit(1)

        for row in cur:
            if self.Debug:
                print("DEFAULT {0} {1}".format(tbl_type, row))

        return

    def dump_rows_name(self, cur=None, hdr=None, tbl_type=None):
        """Dump the contents of the sql cursor for tbl type NAME."""
        if cur is None:
            print("Error dump_rows_name: No cursor specified - exiting.")
            sys.exit(1)

        if hdr is None:
            print("Error dump_rows_name: No col headers specified - exiting.")
            sys.exit(1)

        if tbl_type is None:
            print("Error dump_rows_name: No table type name specified - exiting.")
            sys.exit(1)

        # Get Row indexes
        if ('objectKind' in hdr) and ('objectId' in hdr) and ('name'):
            obj_kind_idx = hdr['objectKind']
            obj_id_idx = hdr['objectId']
            name_idx = hdr['name']
        else:
            print("Error - unexpected col names for tbl type {0} exiting...".format(tbl_type))
            sys.exit(1)

        for row in cur:
            if self.Debug:
                print("{0} {1} {2} {3}".format(tbl_type, row[name_idx], row[obj_kind_idx], row[obj_id_idx]))

        return

    def dump_rows_strings(self, cur=None, hdr=None, tbl_type=None):
        """Dump the contents of the sql cursor for tbl type StringIds."""
        if cur is None:
            print("Error dump_rows_strings: No cursor specified - exiting.")
            sys.exit(1)

        if hdr is None:
            print("Error dump_rows_strings: No col headers specified - exiting.")
            sys.exit(1)

        if tbl_type is None:
            print("Error dump_rows_strings: No table type name specified - exiting.")
            sys.exit(1)

        if ('id' in hdr) and ('value' in hdr):
            str_id_idx = hdr['id']
            str_name_idx = hdr['value']

        for row in cur:
            str_id = row[str_id_idx]
            str_name = row[str_name_idx]
            if str_id not in self.string_hash:
                self.string_hash[str_id] = str_name
            if self.Debug:
                print("{0} {1} {2}".format(tbl_type, row[str_id_idx], row[str_name_idx]))

        return

    def dump_rows_conc_kernel(self, cur=None, hdr=None, tbl_type=None):
        """
        Dump the contents of the sql cursor for tbl type CONCURRENT_KERNEL.
        Note that the correlation ID in conc kernel maps to correlation ID in Runtime Not always true in the reverse
        direction - Runtime covers more events than just kernel.
        """
        # Get Row indexes
        if ('start' in hdr) and ('end' in hdr) and ('registersPerThread' in hdr)\
            and ('demangledName' in hdr) and ('correlationId' in hdr) and ('streamId' in hdr):
            start_idx = hdr['start']
            end_idx = hdr['end']
            corr_id_idx = hdr['correlationId']
            name_id_idx = hdr['demangledName']
            stream_id_idx = hdr['streamId']
            regs_per_th_idx = hdr['registersPerThread']
        else:
            print("Error - unexpected col names for tbl type {0} exiting...".format(tbl_type))
            sys.exit(1)

        if self.Debug:
            print("TblType ElapsedTime(ns) StartTime(ns) EndTime(ns) StreamId CorrId Regs Name")
        for row in cur:
            name_id = row[name_id_idx]
            start_time = row[start_idx]
            end_time = row[end_idx]
            string_name = self.string_hash[name_id]
            ## Get the first time stamp so we can subtract off the time since epoc
            if self.time_base == -1:
                self.time_base = start_time

            if self.Debug:
                time_base = 0
                print("{0} {1} {2} {3} {4} {5} {6} {7}".format(tbl_type, end_time - start_time,
                                                               start_time - time_base, end_time - time_base,
                                                               row[stream_id_idx], row[corr_id_idx],
                                                               row[regs_per_th_idx], string_name))

        return

    def dump_rows_marker(self, cur=None, hdr=None, tbl_type=None):
        """
        Dump the contents of the sql cursor for tbl type MARKER.
        Format for this table is 2 lines per event:
        First row - time stamp is the start time and the 'name' field is the string name of the event
            Use the String Table to lookup the names - name to ID mapping - only valid for start of event row
            The 'id' col is the event ID and it should be the same for both rows
        2nd Row - Time stamp is stop time
            Use 'id' to match up the start time stamp and event info
            Additional info is available in the marker_data() table - use 'id' to lookup this data
            'Category' is the field that is reported by the GUI
            _id_,flags,timestamp,id,objectKind,objectId,name,domain
            1,2,1509565664581882230,1,2,"^Z",3,0
            2,4,1509565664620622854,1,2,"^Z",0,0
        """

        marker_hash = {}
        if cur is None:
            print("Error dump_rows_marker: No cursor specified - exiting.")
            sys.exit(1)

        if hdr is None:
            print("Error dump_rows_marker: No col headers specified - exiting.")
            sys.exit(1)

        if tbl_type is None:
            print("Error dump_rows_marker: No table type name specified - exiting.")
            sys.exit(1)

        # Get Row indexes
        if ('timestamp' in hdr) and ('flags' in hdr) and ('id' in hdr) and ('name' in hdr) and ('objectKind'
                                                                                                in hdr) and ('objectId'
                                                                                                             in hdr):
            ts_idx = hdr['timestamp']
            flag_idx = hdr['flags']
            event_id_idx = hdr['id']
            name_id_idx = hdr['name']
            object_kind_idx = hdr['objectKind']
            object_id_idx = hdr['objectId']
        else:
            raise Exception("Error - unexpected col names for tbl type {0} exiting...".format(tbl_type))

        if self.Debug:
            print(
                "TblType EventId NameId ElapsedTime(ns) StartTime(ns) EndTime(ns) LayerName LayerInstance ObjectKind ProcID ThreadID"
            )
        for row in cur:
            if self.time_base == -1:
                self.time_base = row[ts_idx]
                break
            event_id = row[event_id_idx]
            ## Save the name_id and the start time stamp for each event
            if event_id not in marker_hash:
                marker_hash[event_id] = [row[name_id_idx], row[ts_idx]]
                #print ("Adding event_id {0} to marker hash".format(event_id))
            else:
                name_id, start_time = marker_hash[event_id]
                elapsed_time = row[ts_idx] - start_time  ## Elapsed time in ns
                string_net_name = self.string_hash[name_id]
                net_name = string_net_name
                long_name = ""
                ## Try to figure out the framework that was used to generate the NVTX instrumentation
                if self.FwType is None:
                    self.FwType = detect_fw_type(net_name)

                pat = re.compile(r"(\S+)\s+(\S+)")
                a = re.match(pat, string_net_name)
                if a:
                    net_name = a.group(1)
                    long_name = a.group(2)
                #import pdb;pdb.set_trace()
                ## Convert ObjId into thread ID
                proc_id, thread_id = decode_object_id(row[object_kind_idx], row[object_id_idx])
                if self.Debug:
                    time_base = 0
                    print("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}".format(
                        tbl_type, event_id, name_id, elapsed_time, start_time - time_base,
                        row[ts_idx] - time_base, net_name, long_name, row[object_kind_idx], proc_id, thread_id))
                if (row[flag_idx] != 4):
                    print("Error - unexpected flag {0} for row {1}".format(row[flag_idx], row))
                del (marker_hash[event_id])

        return

    def make_th_id_64bit(self, thread_id):
        """
        For integer values that are < MAX_INT32 and have non zero bit 31 they got converted to negative number by 
        this equation:  value - MAX_INT32 = new_value (negative number)
        This code converts the negative number back to the positive int it is supposed to be : pos_int = neg_int + max_int32
        """
        if thread_id < 0:
            thread_id = NsysParser.MAX_INT32 + thread_id
        return thread_id

    def dump_rows_runtime_driver(self, cur=None, hdr=None, tbl_type=None):
        """
        Dump the contents of the sql cursor for TBL type RUNTIME or driver.
        Walk all the rows in the table and print runtime events map to different tables.
        Many events in runtime are cuda events 
            - use the correlation ID to lookup the CUDA event ID in the the table CUDA_EVENT
        The events are numbered 
            - I don't see a string equivalent to the number
        The profiler must have an internal decoder for these events
            - The other type of event is kernel event
            - These events map to a different table
            - So if the correlation ID is not found in cuda_event table
            - Look in concurrent Kernel event table
            - If the correlation ID matches - then check the Name ID field
            - The name ID should return the string name of the event
            - You can also compare time stamp info because the kernel table tracks it
        """

        if cur is None:
            print("Error dump_rows_runtime_driver: No cursor specified - exiting.")
            sys.exit(1)

        if hdr is None:
            print("Error dump_rows_runtime_driver: No col headers specified - exiting.")
            sys.exit(1)

        if tbl_type is None:
            print("Error dump_rows_runtime_driver: No tbl type name specified - exiting.")
            sys.exit(1)

        # Get start time stamp
        if ('start' in hdr) and ('end' in hdr) and ('globalTid' in hdr) and ('correlationId' in hdr)\
            and ('returnValue' in hdr):
            start_idx = hdr['start']
            end_idx = hdr['end']
            thread_idx = hdr['globalTid']
            corr_idx = hdr['correlationId']
            ret_val_idx = hdr['returnValue']
        else:
            print("Error: Col Hdrs {}", format(hdr))
            sys.exit(1)

        # Walk the cursor - print each row
        if self.Debug:
            print("Start_time(ns) End_time(ns) Elapsed_time(ns) Thread_id Correlation_id RetVal_id")
        num_rows = 0
        for row in cur:
            num_rows += 1
            ## Use driver to set up start time val
            if self.time_base == -1 and tbl_type == 'DRIVER':
                self.time_base = row[start_idx]
                #break
            thread = row[thread_idx]
            # For integer values that are < MAX_INT32 and have non zero bit 31 they got converted to negative number by
            thread = self.make_th_id_64bit(thread)
            if self.Debug:
                time_base = 0
                #print ("{0} {1} {2} {3} {4} {5} {6}".format(tbl_type, row[start_idx]-time_base,\
                #row[end_idx] - time_base, row[end_idx] - row[start_idx], thread, row[corr_idx],\
                #row[ret_val_idx]))
                print("{0} {1} {2} {3} {4} {5} {6}".format(tbl_type, row[start_idx]-time_base,\
                                                           row[end_idx]-time_base, row[end_idx] - row[start_idx],\
                                                           thread, row[corr_idx],\
                                                           row[ret_val_idx]))

        ## Error checking - Driver and RUNTIME must be non empty
        if num_rows == 0:
            pass
            #raise Exception ("Table {} is empty".format(tbl_type))
        return

    def get_tbl_names(self, cur=None):
        """
        Dump the contents of the sql cursor.
        Walk all the rows in the table and print.
        """
        tbl_list = []
        if cur is None:
            print("Error get_tbl_names: No cursor specified - exiting.\n")
            sys.exit(1)
        for row in cur:
            tbl_name = row[0]
            if self.Debug:
                print("Tbl Name {0:s}".format(tbl_name))
            tbl_list.append(tbl_name)

        return tbl_list

    def process_tbl(self, tbl=None, cur=None, name=None):
        """Decode the DRIVER table"""
        if tbl is None:
            print("-Error- process_tbl: No tbl specified - exiting.\n")
            sys.exit(1)
        if cur is None:
            print("-Error- process_tbl: No cursor specified - exiting.\n")
            sys.exit(1)
        if name is None:
            print("-Error- process_tbl: No name specified - exiting.\n")
            sys.exit(1)

        pattern = re.compile(name)
        if pattern.search(tbl):
            cmd_string = "select * from {};".format(tbl)
            if self.Debug:
                print("Executing sql cmd {}".format(cmd_string))
            cur.execute(cmd_string)  ## Need to use a tuple for variable sub- even though only passing 1 value
            tbl_hdr = get_tbl_hdrs(cur, self.Debug)
            self.dump_rows(cur, tbl_hdr, name)

    def get_marker_pandas_tbl_frame(self, tbl=None, cur=None):
        """Returns pandas tbl frame for the marker table."""
        query_string = "select text, start, end, globalTid from {}".format(tbl)
        tbl_hash = {
            'name': [],
            'start_time': [],
            'end_time': [],
            'total_time': [],
            'proc_id': [],
            'thread_id': []
        }  ## Hash used to create pandas frame
        cur.execute(query_string)
        tbl_list = cur.fetchall()
        tbl_hdr = get_tbl_hdrs(cur, False)
        marker_id = tbl_hdr['text']
        start_st = tbl_hdr['start']
        end_st = tbl_hdr['end']
        globalTid = tbl_hdr['globalTid']
        row_cnt = 0

        for row in tbl_list:
            start_time = row[start_st]
            end_time = row[end_st]
            proc_id = int(row[globalTid] / 0x1000000) % 0x1000000
            thread_id = int(row[globalTid] % 0x1000000)
            marker_string = row[marker_id]
            if end_time is None or start_time is None:
                continue
            elapsed_time = end_time - start_time

            tbl_hash['name'].append(marker_string)
            tbl_hash['start_time'].append(start_time)
            tbl_hash['end_time'].append(end_time)
            tbl_hash['total_time'].append(elapsed_time)
            tbl_hash['thread_id'].append(thread_id)
            tbl_hash['proc_id'].append(proc_id)
            row_cnt += 1

        ## Only create Pandas frame if row count is non zero
        if row_cnt > 0:
            panda_frame = pd.DataFrame(tbl_hash)
        else:
            panda_frame = None

        del tbl_hash

        return panda_frame

    def get_runtime_pandas_tbl_frame(self, tbl=None, cur=None):
        """Copy a sql TBL into Pandas frame."""
        tbl_hash = {'start': [], 'end': [], 'threadId': [], 'correlationId': []}
        query_string = "select start, end, globalTid, correlationId from {} ".format(tbl)
        cur.execute(query_string)
        tbl_list = cur.fetchall()
        tbl_hdr = get_tbl_hdrs(cur, False)
        start_idx = tbl_hdr['start']
        end_idx = tbl_hdr['end']
        th_idx = tbl_hdr['globalTid']
        cor_idx = tbl_hdr['correlationId']
        for row in tbl_list:
            thread_id = self.make_th_id_64bit(row[th_idx])
            tbl_hash['start'].append(row[start_idx])
            tbl_hash['end'].append(row[end_idx])
            tbl_hash['threadId'].append(thread_id)
            tbl_hash['correlationId'].append(row[cor_idx])

        panda_frame = pd.DataFrame(tbl_hash)
        del tbl_hash
        return panda_frame

    def getEncapsulatingMarkers(self, db_handle, objId, startTime, endTime):
        """
        getEncapsulatingMarkers()
            Finds all the markers that contain this time range passed in as arguments
        """
        marker_list = []
        markerT     = 'NVTX_EVENTS'
        #Find all encapsulating markers
        cmd = 'SELECT text from {} where \
            globalTid = {} and \
            start < {} and \
            end > {} \
            ORDER BY start ASC'.format(markerT, objId, startTime, endTime)
        result = db_handle.execute(cmd)
        for row in result:
            ## row is a 1 element tuple - we just want the text not the tuple
            marker_list.append(row[0])

        return marker_list

    def link_kernel_to_dl_layer(self, cur=None, tbl_list=None, db_name=None, file_des=None):
        """ 
        Walks the list of GPU kernel events and maps them to user level layer names defined in CPU CUDA runtime threads.
        """

        if cur is None or tbl_list is None or db_name is None or file_des is None:
            print("Error link_kernel_to_dl_layer: bad arguments - exiting.\n")
            sys.exit(1)

        kernel_events = []  ## Empty list - used to store the entire kernel tbl
        ns_to_ms_factor = 1000000
        db_name = os.path.basename(db_file)
        pivot_tbl_tag = re.sub(r'[.]\w+', '', db_name)

        # this query works for runs with CUDA graphs also
        # in CUDA graphs, the correlationId for kernels launched from a graph 
        # are all the same, so the additional GROUP BY CUPTI_ACTIVITY_KIND_KERNEL.start is necessary
        nvtx_text_id = 'NVTX_EVENTS.jsonTextId'
        if self.FwType == 'FW_TENSOR_FLOW':
            nvtx_text_id = 'NVTX_EVENTS.textId'
        query_string = '''SELECT 
                                CUPTI_ACTIVITY_KIND_RUNTIME.correlationId, 
                       '''
        query_string = query_string + nvtx_text_id + ','
        query_string = query_string + \
                       '''
                                NVTX_EVENTS.text,
                                NVTX_EVENTS.globalTid, 
                                MAX(NVTX_EVENTS.start), 
                                NVTX_EVENTS.end, 
                                CUPTI_ACTIVITY_KIND_KERNEL.demangledName, 
                                CUPTI_ACTIVITY_KIND_KERNEL.start, 
                                CUPTI_ACTIVITY_KIND_KERNEL.end, 
                                CUPTI_ACTIVITY_KIND_KERNEL.gridX, 
                                CUPTI_ACTIVITY_KIND_KERNEL.gridY, 
                                CUPTI_ACTIVITY_KIND_KERNEL.gridZ, 
                                CUPTI_ACTIVITY_KIND_KERNEL.blockX, 
                                CUPTI_ACTIVITY_KIND_KERNEL.blockY, 
                                CUPTI_ACTIVITY_KIND_KERNEL.blockZ 
                            FROM 
                                NVTX_EVENTS 
                            JOIN 
                                CUPTI_ACTIVITY_KIND_RUNTIME 
                            ON 
                                (NVTX_EVENTS.eventType == 59 OR NVTX_EVENTS.eventType == 60)
                                AND NVTX_EVENTS.globalTid == CUPTI_ACTIVITY_KIND_RUNTIME.globalTid
                                AND NVTX_EVENTS.start <= CUPTI_ACTIVITY_KIND_RUNTIME.start 
                                AND NVTX_EVENTS.end >= CUPTI_ACTIVITY_KIND_RUNTIME.end 
                            JOIN 
                                CUPTI_ACTIVITY_KIND_KERNEL 
                            ON 
                                CUPTI_ACTIVITY_KIND_RUNTIME.correlationId  == CUPTI_ACTIVITY_KIND_KERNEL.correlationId
							GROUP BY 
                                NVTX_EVENTS.globalTid, 
                                CUPTI_ACTIVITY_KIND_RUNTIME.correlationId, 
                                CUPTI_ACTIVITY_KIND_KERNEL.start                            
                            '''
        # First check if there is cuda graph kernel exist
        #is_cuda_graph_kernel = '''
        #                       SELECT correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL
        #                       WHERE graphNodeId
        #                       '''

        print('Reading from sqlite')
        start = time.time()
        print('Querying kernels (can take a long time)...')
        cur.execute(query_string)
        kernel_events = cur.fetchall()
        end = time.time()
        print('query exec time: {} sec'.format(int((end - start))))

        ## Store the table in a dict - Col headers are the keys - each val is a list, then pass the dict to Pandas to make
        ## a frame Walk each row in the table - query the RUNTIME tbl for CPU start/end times
        report_tbl = {
            'LayerName': [],
            'LayerOpName': [],
            'LayerType': [],
            'TensorShapes': [],
            'Phase': [],
            'TensorShapes': [],
            'PhaseConfidence': [],
            'CPUStartTime(ms)': [],
            'CPUEndTime(ms)': [],
            'CPUDuration(ms)': [],
            'GPUStartTime(ms)': [],
            'GPUEndTime(ms)': [],
            'GPUDuration(ms)': [],
            'CorrId': [],
            'Thread': [],
            'Kernel': [],
            'ExperTag': [],
            'GridXYZ': [],
            'BlockXYZ': []
        }
        print(
            "LayerName|LayerType|TensorShapes|Phase|PhaseConfidence|CPUStartTime(ms)|CPUEndTime(ms)|CPUDuration(ms)|GPUStartTime(ms)|GPUEndTime(ms)|GPUDuration(ms)|CorrId|Thread|Kernel|ExperTag|GridXYZ|BlockXYZ",
            file=file_des)
        if self.export_trt_json:
            json_filer = open(self.json_file_name, 'w')
            layersjson = {}
            layersjson['Layers'] = []

        start = time.time()
        for kernel in kernel_events:
            [
                corr_id, layer_text_id, text, GTid, marker_start, marker_end, ker_name_id, start_time, end_time, grid_x,
                grid_y, grid_z, block_x, block_y, block_z
            ] = kernel
            ## Need to call map from name_id to name
            #mangled_ker_name = self.string_hash[ker_name_id]
            #ker_name = self.demangle_kernel_name(mangled_ker_name)
            ## Nsight already contains demangled kernel names
            ker_name = self.string_hash[ker_name_id]

            grid_coords = "Grid-{}-{}-{}".format(grid_x, grid_y, grid_z)
            block_coords = "Block-{}-{}-{}".format(block_x, block_y, block_z)

            # Use correlation ID to map kernel event to runtime cpu event
            thread_id = GTid % 0x1000000

            ## Here the 2 fields from marker name mean different things depending on the framework
            ## For caffe2 the first field is the general layer name, the 2nd field is the layer instance
            ## For SFW (custome FW) - the first field is the phase naem (fprop/dgrad/wgrad) and 2nd
            ## is the layer instance name
            if layer_text_id and self.FwType != "FW_MXNET":
                marker_name = self.string_hash[layer_text_id]
            elif text:
                marker_name = text
            else:
                from netex.network_exporters.exporter_utils import log_warn
                log_warn("No NVTX_EVENTS.jsonTextId or NVTX_EVENTS.text found for {}, skip.".format(ker_name))
                continue
            layer_instance = marker_name
            marker_stack = []
            if self.FwType == 'FW_PYTORCH':
                marker_stack = self.getEncapsulatingMarkers(cur, GTid, marker_start, marker_end)
            ## Use tatic name as kernel name if possible, it is equivalent of shader name but more human readable.
            ## Especially for XMMA higly template based kernel.
            ker_name = self.decode_trt_kernel_name(layer_instance) or ker_name
            ###  @@@ Move this code into a function that figures out which fw thn returns type, phase,
            ## name
            if self.export_trt_json:
                layer_type, phase, layer_instance, layer_op, phase_confidence, layer_tensors = self.decode_nvtx_marker_layer_string(
                    layer_instance, layersjson=layersjson)
            else:
                layer_type, phase, layer_instance, layer_op, phase_confidence, layer_tensors =\
                    self.decode_nvtx_marker_layer_string(layer_instance, kernel_name=ker_name, marker_stack=marker_stack)

            # A check for MXNet add_dgrad
            self.mxnet_check_add_dgrad(layer_instance, report_tbl)

            ## Some thread IDs use the most significant bit in signed int32
            ## This converts the value to a positive integer
            if thread_id < 0:
                thread_id = NsysParser.MAX_INT32 + thread_id

            ## Convert time units ns -> ms
            marker_start_ms = time_stamp_to_duration(marker_start, self.time_base, ns_to_ms_factor)
            marker_end_ms = time_stamp_to_duration(marker_end, self.time_base, ns_to_ms_factor)
            marker_time_ms = time_stamp_to_duration(marker_end, marker_start, ns_to_ms_factor)
            gpu_start_ms = time_stamp_to_duration(start_time, self.time_base, ns_to_ms_factor)
            gpu_end_ms = time_stamp_to_duration(end_time, self.time_base, ns_to_ms_factor)
            gpu_time_ms = time_stamp_to_duration(end_time, start_time, ns_to_ms_factor)

            th_id_str = "th_{}".format(thread_id)
            #print("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|".format(layer_instance, layer_type, layer_tensors, phase, marker_start_ms, marker_end_ms, marker_time_ms, gpu_start_ms , gpu_end_ms, gpu_time_ms, corr_id, th_id_str, ker_name, pivot_tbl_tag, grid_coords, block_coords), file=file_des)
            report_tbl['LayerName'].append(layer_instance)
            report_tbl['LayerOpName'].append(layer_op)
            report_tbl['LayerType'].append(layer_type)
            report_tbl['TensorShapes'].append(layer_tensors)
            report_tbl['Phase'].append(phase)
            report_tbl['PhaseConfidence'].append(phase_confidence)
            report_tbl['CPUStartTime(ms)'].append(marker_start_ms)
            report_tbl['CPUEndTime(ms)'].append(marker_end_ms)
            report_tbl['CPUDuration(ms)'].append(marker_time_ms)
            report_tbl['GPUStartTime(ms)'].append(gpu_start_ms)
            report_tbl['GPUEndTime(ms)'].append(gpu_end_ms)
            report_tbl['GPUDuration(ms)'].append(gpu_time_ms)
            report_tbl['CorrId'].append(corr_id)
            report_tbl['Thread'].append(th_id_str)
            report_tbl['Kernel'].append(ker_name)
            report_tbl['ExperTag'].append(pivot_tbl_tag)
            report_tbl['GridXYZ'].append(grid_coords)
            report_tbl['BlockXYZ'].append(block_coords)

        end = time.time()
        print('report tbl gen time: {} sec'.format(int((end - start))))
        if self.export_trt_json:
            json.dump(layersjson, json_filer, indent=2)
            json_filer.close()
        print("Finished processing SQLITE file, creating Pandas Frame")
        ## Create Pandas data frame
        data_frame = pd.DataFrame(report_tbl)
        del report_tbl

        return data_frame

    def compute_ave_runtime(self, pd_frame):
        """
        count unique occurances of LayerName/Phase/Kernel - get the average of GPUDuration for each one.
        Return a new frame that just has GPUDuration Layer Info / Exper / Grid...
        """
        ## foreach layer_name
        ##  filter - on layer_name & Phase
        ##    returns a table / frame - now need to filter on kernel name
        ##  Now you should have a frame that contains exactly the same number of rows as the
        ##  number of iterations
        ##  Use data_frame['col_name'].mean() function to get the mean runtime
        ##  Create a new DF row using same col headings as old frame - take out the CPU start / end & GPU start end
        ##    Get row[0] from the frame used to compute mean - use all the other field values
        ##    Now you have a table w/ layer_name / layer_type / kernel / phase / Time (ms)
        ave_val_tbl = {
            'LayerName': [],
            'LayerOpName': [],
            'LayerType': [],
            'Phase': [],
            'TensorShapes': [],
            'PhaseConfidence': [],
            'CPUDuration(ms)': [],
            'GPUDuration(ms)': [],
            'CorrId': [],
            'Thread': [],
            'Kernel': [],
            'ExperTag': [],
            'GridXYZ': [],
            'BlockXYZ': []
        }
        unique_layer_names = get_unique_tags_from_frame('LayerName', pd_frame)
        unique_phase_names = get_unique_tags_from_frame('Phase', pd_frame)

        if self.Debug:
            print("Found {} unique layers ".format(len(unique_layer_names)))
        ## Now foreach unique layer_name - extract a frame that only has data for that layer name
        for layer_name in unique_layer_names:
            tmp_frame = pd_frame.loc[pd_frame['LayerName'] == layer_name]
            for phase in unique_phase_names:
                ## Print all rows in where col 'LayerName' == layer_name phase == Phase
                ## Break it into 2 searches - first get all the layer names - then search for phase
                ## in a much smaller frame
                ## @@@ What if I throw out the first row?  That should get rid of cudnn_find
                layer_frame = tmp_frame.loc[tmp_frame['Phase'] == phase]
                ## Some layers are only available in some phases
                if (layer_frame.empty):
                    continue

                unique_cpu_start = get_unique_tags_from_frame('CPUStartTime(ms)', layer_frame)
                loop_num = len(unique_cpu_start)
                #print("layer_frame {}".format(layer_frame))
                ## @@@ Almost done - if there is more than 1 kernel name - filter based on kernel
                ## names used in this layer
                unique_kernel_names = get_unique_tags_from_frame('Kernel', layer_frame)

                # Handle layers with mult-kernel(like LSTM layers)
                # Calculate layer duration instead of every kernel duration for DLSim corelation
                if len(unique_kernel_names) > 1:
                    layer_cpu_runtime = 0
                    layer_gpu_runtime = 0
                    for cpu_start_time in unique_cpu_start:
                        cpu_frame =  layer_frame.loc[layer_frame['CPUStartTime(ms)'] == cpu_start_time]
                        layer_cpu_runtime += cpu_frame['CPUDuration(ms)'].mean()
                        #layer_gpu_runtime += cpu_frame['GPUEndTime(ms)'].max() - cpu_frame['GPUStartTime(ms)'].min()
                        layer_gpu_runtime += cpu_frame['GPUDuration(ms)'].sum()
                    first_row_values = layer_frame.iloc[0]
                    ave_val_tbl['GPUDuration(ms)'].append(layer_gpu_runtime / loop_num)
                    ave_val_tbl['CPUDuration(ms)'].append(layer_cpu_runtime / loop_num)
                    ave_val_tbl['Kernel'].append(layer_name + "_total")
                    ave_val_tbl['LayerName'].append(layer_name)
                    ave_val_tbl['Phase'].append(phase)
                    ave_val_tbl['TensorShapes'].append(first_row_values['TensorShapes'])
                    ave_val_tbl['PhaseConfidence'].append(first_row_values['PhaseConfidence'])
                    ave_val_tbl['BlockXYZ'].append(first_row_values['BlockXYZ'])
                    ave_val_tbl['CorrId'].append(first_row_values['CorrId'])
                    ave_val_tbl['ExperTag'].append(first_row_values['ExperTag'])
                    ave_val_tbl['GridXYZ'].append(first_row_values['GridXYZ'])
                    ave_val_tbl['LayerType'].append(first_row_values['LayerType'])
                    ave_val_tbl['LayerOpName'].append(first_row_values['LayerOpName'])
                    ave_val_tbl['Thread'].append(first_row_values['Thread'])
                else:
                    for kernel in unique_kernel_names:
                        kernel_frame = layer_frame.loc[layer_frame['Kernel'] == kernel]
                        ## Exclude the last row if the grid size does not match
                        ## (This is caused by the fact that last run might not use max batch size/max seq length)
                        id_max_batch = kernel_frame['GPUDuration(ms)'].idxmax()
                        GridXYZ = kernel_frame['GridXYZ'].loc[id_max_batch]
                        BlockXYZ = kernel_frame['BlockXYZ'].loc[id_max_batch]
                        kernel_frame = kernel_frame[kernel_frame['GridXYZ'] == GridXYZ]
                        kernel_frame = kernel_frame[kernel_frame['BlockXYZ'] == BlockXYZ]
                        ave_gpu_runtime = kernel_frame['GPUDuration(ms)'].mean()
                        ave_cpu_runtime = kernel_frame['CPUDuration(ms)'].mean()
                        first_row_values = kernel_frame.iloc[0]
                        ## Now start building a new frame from a dictionary
                        ave_val_tbl['GPUDuration(ms)'].append(ave_gpu_runtime)
                        ave_val_tbl['CPUDuration(ms)'].append(ave_cpu_runtime)
                        ave_val_tbl['Kernel'].append(kernel)
                        ave_val_tbl['LayerName'].append(layer_name)

                        if self.FwType == 'FW_MXNET':
                            if kernel.find("scalePackedTensor") >= 0:
                                ave_val_tbl['Phase'].append('wgrad')
                            elif kernel.find("convertTensor") >= 0:
                                ave_val_tbl['Phase'].append('wgrad')
                            elif kernel.find("computeOffsetsKernel") >= 0:
                                ave_val_tbl['Phase'].append('dgrad')
                            elif kernel.find("computeBOffsetsKernel") >= 0:
                                ave_val_tbl['Phase'].append('dgrad')
                            else:
                                ave_val_tbl['Phase'].append(phase)
                        else:
                            ave_val_tbl['Phase'].append(phase)
                        ave_val_tbl['PhaseConfidence'].append(first_row_values['PhaseConfidence'])
                        ave_val_tbl['TensorShapes'].append(first_row_values['TensorShapes'])
                        ave_val_tbl['BlockXYZ'].append(first_row_values['BlockXYZ'])
                        ave_val_tbl['CorrId'].append(first_row_values['CorrId'])
                        ave_val_tbl['ExperTag'].append(first_row_values['ExperTag'])
                        ave_val_tbl['GridXYZ'].append(first_row_values['GridXYZ'])
                        ave_val_tbl['LayerType'].append(first_row_values['LayerType'])
                        ave_val_tbl['LayerOpName'].append(first_row_values['LayerOpName'])
                        ave_val_tbl['Thread'].append(first_row_values['Thread'])
                        if self.Debug:
                            print("Layer {} Phase {} Kernel {} GPU-Time {} CPU-Time {} iterations {}".format(
                                layer_name, phase, kernel, ave_gpu_runtime, ave_cpu_runtime, loop_num))

        ## Now put everything into a frame
        ave_val_frame = pd.DataFrame.from_dict(ave_val_tbl)
        return ave_val_frame

    def strip_layer_name(self, l_name):
        '''
        Strip the preamble off of a layer name 
        Return stripped down name and the part that
        was removed
        '''
        l_inst_tag = l_name
        l_inst = l_name
        ## Look for eg.  conv2_1_2_bn, conv1_3_shortcut_bn, conv1_bn
        pattern = re.compile(r"(conv\d+_\d+_\d+|conv\d+_\d+|conv\d+)_(\S+)")
        res = re.search(pattern, l_name)
        if res is not None:
            l_inst_tag = res.group(1)
            l_inst = res.group(2)
            if self.Debug:
                print("Layer -> {} Layer inst tag -> {} - layer inst -> {}".format(l_name, l_inst_tag, l_inst, l_name))

        return l_inst, l_inst_tag

    def get_pytorch_layer_type_from_name(self, layer_instance):
        l_name = layer_instance
        l_phase = "Fprop"

        ## Remove the pytorch tag eg N5torch8autograd3
        l_name = re.sub(r"^N\d+\w+\d+\w+\d+", "", l_name)
        l_type = l_name

        pat = re.compile(r"(\w+)Forward")
        res = re.search(pat, l_name)
        if res is not None:
            l_type = res.group(1)
            l_phase = "Fprop"
            return [l_type, l_phase, l_name]

        pat = re.compile(r"(\w+)GradE")
        res = re.search(pat, l_name)
        if res is not None:
            l_type = res.group(1)
            l_phase = "Dgrad"
            return [l_type, l_phase, l_name]

        pat = re.compile(r"(\w+)[Bb]ackward[E]*$")
        res = re.search(pat, l_name)
        if res is not None:
            l_type = res.group(1)
            l_phase = "Wgrad"
            return [l_type, l_phase, l_name]

        return [l_type, l_phase, l_name]

    ## Find the mapping between fusion kernel names and the 
    ## TF layers that it is made of
    def map_xla_kernel_to_tf_op(self, l_name, kernel_name):
        kernel_map = None
        l_join_str = " + "
        l_names = []
        l_types = []
        if re.match(r"fusion", kernel_name):
            kernel_name = re.sub("_", ".", kernel_name)
            cluster_name = re.sub(r"^(cluster_\d+)\S*$", '\g<1>', l_name)
            if cluster_name in self.xla_kernel_map:
                xla_kernel_info = self.xla_kernel_map[cluster_name]
                if kernel_name in xla_kernel_info:
                    kernel_map = xla_kernel_info[kernel_name]
                    for layer, op_type in  kernel_map.items():
                        if layer not in l_names:
                            l_names.append(layer)
                        if op_type not in l_types:
                            l_types.append(op_type)
                    if self.Debug:
                        print("Fused kernel {} maps to layer {} op {}".format(kernel_name, l_names, l_types))

        l_type = l_join_str.join(sorted(l_types))
        fused_l_name = l_join_str.join(sorted(l_names))
        return fused_l_name, l_type

    def tf_check_backward_op(self, l_name, l_type, unique_name):
        l_name = clean_conditional_autograd_lname(l_name, l_type)
        grad_pattern = re.search(r"(?:gradients/)*gradients[_1-9]*[/](\S+)", l_name)
        momentum_pat = re.search(r"(Momentum|Adam|RMSProp).update_(\S+)", l_name)
        l_phase = ""
        if grad_pattern:
            ## At this point it could be either wgrad or dgrad - default to wgrad
            ## wgrad includes ConvBackpropFilter and ops like FusedBatchNormGrad
            ## Dgrad is only for ops that use the chain rule and split weights/data gradient
            short_l_name = grad_pattern.group(1)
            ## l_name needs to match the forward pass name of the same layer
            l_name = short_l_name
            l_phase = "Dgrad"
            if re.search(r"BackpropFilter", l_type):
                l_phase = "Wgrad"
            else:
                print("Op {} Dgrad".format(unique_name))
            l_type = re.sub("Backprop.*$", "", l_type)
            ## Also remove '_grad/Conv[1-3]DBackprop' from end of layer name
            ## eg Conv2DBackpropFilter -> Conv2D
            l_name = re.sub("_grad/(\S+)Backprop.*$", "", l_name)
            l_type = re.sub("Grad.*$", "", l_type)

            ## Some generated ops have this pattern
            ## MatMul_grad/MatMul or MatMul_grad/MatMul_1
            ## Look for the pattern before _grad - if you see it a 2nd time
            ## remove the pattern_grad/
            orig_name = l_name
            pat = re.search("([A-Za-z]+)([_0-9]*)_grad/", l_name)
            if pat:
                repeat_pat        = pat.group(1)
                repeat_with_upper = repeat_pat[0].upper() + repeat_pat[1:]
                patterns_to_try_list = [repeat_pat, repeat_with_upper]
                op_pat            = pat.group(2)
                full_op_pat       = repeat_pat
                if op_pat:
                    full_op_pat = repeat_pat + op_pat
                l_name_test = l_name
                for repeat_pat in patterns_to_try_list:
                    ## eg mul_1_grad/Mul_1  -> mul_1
                    ## MatMul_grad/MatMul_1 -> MatMul
                    #l_name = re.sub("{}_grad/{}_1".format(full_op_pat, repeat_pat), full_op_pat, l_name_test)
                    #l_name = re.sub("{}_grad/[A-Za-z]+[_1]*".format(full_op_pat), full_op_pat, l_name_test)
                    ## The _1 at the end of the op name is how we determine that the op is wgrad
                    ## The _1 means that the op was generated, it is the 2nd instance of a generated
                    ## op related to a specific fwd pass op
                    l_name = re.sub("{}_grad/[A-Za-z]+_1*".format(full_op_pat), full_op_pat, l_name_test)
                    first_sub = l_name
                    ## If sub worked - mark this op as Wgrad
                    if (l_name != l_name_test):
                        ## Autogenerated Wgrad ops have _1 at the end
                        ## of the name, that's what distunguishes them from dgrad op
                        if re.search('BatchMatMul', l_type):
                            l_phase = "Dgrad"
                        else:
                            l_phase = "Wgrad"
                    else:
                        if full_op_pat == "BiasAdd" or re.search("lookup|embedding|Gather", repeat_pat):
                            ## BiasAdd_grad is actually wgrad, not dgrad
                            l_phase = "Wgrad"
                        else:
                            ## eg mul_1_grad/Mul  -> mul1
                            ## MatMul_grad/MatMul -> MatMul
                            l_phase = "Dgrad"
                        ## This sub covers layers that do not have the trailing _1 in the name
                        l_name = re.sub("{}_grad/[A-Za-z0-9]+".format(full_op_pat), full_op_pat, l_name_test)
                    if l_name != l_name_test:
                        break

            ## Now look for a similar pattern
            ## eg Softmax_grad/mul  -> Softmax
            ## eg Softmax_grad/mul_1  -> Softmax_1
            #l_name = re.sub(r"_grad/\w+[_0-9]*$", "", l_name)

        elif re.search(r"train.update_model", l_name):
            l_phase = "weight_update"
        elif  momentum_pat:
            l_phase = "weight_update"
            weight_name =  momentum_pat.group(2)
            var_name = re.sub(r"/\w*Apply\w+", "", weight_name)
            if var_name in self.weight_to_fprop_map:
                l_name = self.weight_to_fprop_map[var_name]['fprop_layer']
                l_type = self.weight_to_fprop_map[var_name]['fprop_op']
            else:
                l_name = var_name


        #elif re.search(r"(cross_entropy|Loss|Momentum)", unique_name):
        elif re.search(r"cross_entropy", unique_name):
            l_phase = "Wgrad"


        if not l_phase:
            l_phase = "Fprop"

        return l_type, l_phase, l_name

    ################################################################################
    ## get_tensorflow_layer_info_with_map() 
    ##   
    ################################################################################
    def get_tensorflow_layer_info_with_map(self, layer_instance, kernel_name=None):
        '''
        Tensorflow graph groups operators into clusters
        The graph_info_map maps the unique operator name to a cluster name
        If there is no cluster associated with the operator name - then just default to 
        get_tensorflow_layer_type_from_name()
        '''
        
        # In TF2 gradients are stored/watched using gradient_tape and the string is 
        # is prepended to gradient operations. 
        # Replacing 'gradient_tape/' with 'gradients/' makes the format compliant
        # with rest of the instrumentation in this script
        layer_instance = layer_instance.replace("gradient_tape/", "gradients/")

        l_type       = layer_instance
        l_name       = l_type
        op_name      = None
        unique_name  = layer_instance

        # For example "RealDiv: truediv" or "Cast: Cast_1"    
        res = re.search(r"(\S+):\s+(\S+)", layer_instance)
        if res :
            op_name     = res.group(1)
            unique_name = res.group(2)
        else :
            raise Exception("Unexpected NVTX marker format {} for Tensorflow framework".format(layer_instance))

        unique_name = clean_conditional_layer_name(unique_name)
        l_type    = op_name
        l_name    = unique_name
        l_op_name = unique_name
        ## If unique layer name not found in graph_info_map
        ## then it is an automatically generated backward pass op
        is_autograd_op = True
        l_phase        = "weight_update"

        is_fused_xla_op = False
        ## XLA requires a mapping table to match kernels to original TF ops
        if self.xla_enable:
            l_phase        = "Fprop"
            op_name, op_type = self.map_xla_kernel_to_tf_op(l_name, kernel_name)
            if op_name and op_type:
                l_op_name = op_name
                l_type = op_type
                is_fused_xla_op = True

        else:
            ## If layer name found in graph_info_map then it is fwd pass
            ## by definition due to dnnx graph export stage
            if unique_name in self.graph_info_map:
                cluster = self.graph_info_map[unique_name]
                l_name  = cluster
                l_phase        = "Fprop"
                is_autograd_op = False
                #print("Layer instance {} is in cluster {}".format(unique_name, cluster))

        if is_autograd_op:
            if is_fused_xla_op:
                unique_name = l_op_name
                l_type, l_phase, _      = self.tf_check_backward_op(unique_name, l_type, unique_name)
            else:
                l_type, l_phase, l_name = self.tf_check_backward_op(l_name, l_type, unique_name)

        return [l_type, l_phase, l_name, l_op_name]

    def get_tensorflow_layer_type_from_name(self, layer_instance):
        '''
        tensorflow layer instance is a '/' separated string like this
        transformer/parallel_0_5/transformer/symbol_modality_33952_512_2/shared/Reshape_1/shape/2
        '''


        l_type       = layer_instance
        l_name       = l_type
        op_name      = None
        unique_name  = layer_instance

        res = re.search(r"(\S+):\s+(\S+)", layer_instance)
        if res :
            op_name     = res.group(1)
            unique_name = res.group(2)
        else :
            raise Exception("Unexpected NVTX marker format {} for Tensorflow framework".format(layer_instance))

        l_type    = op_name
        l_name    = unique_name
        l_op_name = unique_name
        l_phase = "Fprop"


        grad_type = None

        ## Look for back prop layers key word 'training'
        pat = re.compile(r"training[/](\S+)")
        res = re.match(pat, layer_instance)
        if res is not None:
            l_name = res.group(1)
            l_type = l_name
            ## Set phase to Wgrad - dgrad is indicated by the string 'gradient'
            l_phase = "Wgrad"
            ## First - detect whether or not backprop
            ## Take the layer name string - and widdle it down
            ## Optional string between gradients and _grad
            pat = re.compile(r"gradients[/]*(\S*[/]+(\w+)_grad[/](\w+))")
            res = re.search(pat, layer_instance)
            if res is not None:
                l_name = res.group(1)
                grad_type = res.group(2)
                l_type = res.group(3)
                l_phase = "Dgrad"
                l_type = "{}_{}".format(grad_type, l_type)

        return [l_type, l_phase, l_name]

    def get_caffe2_layer_type_from_name(self, layer_operator=None, layer_instance=None):
        '''
        caffe2 has 2 fields to describe the layer - these fields have 3 things
        encoded in the names
        1. Layer instance name
        2. Layer type
        3. Phase - Fprop / Dgrad / Wgrad
        4. Other - weight initialization algo etc
        Under layer type - include param initialization - ParamInit
        Eg BatchNorm -> riv -> running_inv_var, bn_rm ->running_mean, bn_b -> bias bn_s -> scale
        Just make a layer type named Weight Init
        Or - use ConstantFill / MSRAFill to indicate weight init
        '''
        l_type = layer_operator
        l_phase = layer_operator
        l_name = layer_instance

        if layer_operator is None or layer_instance is None:
            print("Error get_caffe2_layer_type_from_name - Bad args - exiting...")
            sys.exit(1)

        pattern = re.compile(r"(\w+)_w_grad")
        res = re.search(pattern, layer_instance)
        extra = "NA"
        if res is not None:
            l_name = res.group(1)
            l_name, l_tag = self.strip_layer_name(l_name)
            l_phase = "Wgrad"
            ## First looks for FilterGradient eg ConvFilterGradient, then looks for just Gradient eg FCGradient
            res = re.search(r"(\w+)(Gradient)", layer_operator)
            if res is not None:
                l_type = res.group(1)
                extra = res.group(2)
                res = re.search(r"(\w+)Filter", l_type)
                if res is not None:
                    l_type = res.group(1)
            if self.Debug:
                print("Found {} phase for layer_instance {} layer_name {} layer_type {} extra {} ".format(
                    l_phase, layer_instance, l_name, l_type, extra))
            return [l_type, l_phase, l_name]

        ## @@@ What about conv_bn ConvDataGradient conv5_3_1_bn_grad - Should layer type be batch norm?
        pattern = re.compile(r"(\w+)_grad")
        res = re.search(pattern, layer_instance)
        if res is not None:
            l_name = res.group(1)
            l_name, l_tag = self.strip_layer_name(l_name)
            l_phase = "Dgrad"
            ## First looks for DataGradient eg ConvDataGradient, then looks for just Gradient eg FCGradient
            res = re.search(r"(\w+)(Gradient)", layer_operator)
            if res is not None:
                l_type = res.group(1)
                extra = res.group(2)
                res = re.search(r"(\w+)Data", l_type)
                if res is not None:
                    l_type = res.group(1)
            if self.Debug:
                print("Found {} phase for layer_instance {} layer_name {} layer_type {} extra {}".format(
                    l_phase, layer_instance, l_name, l_type, extra))

            return [l_type, l_phase, l_name]

        ## Fprop - any layer that isn't a gradient layer
        l_name = layer_instance
        l_phase = "Fprop"
        l_type = layer_operator
        if self.Debug:
            print("Found {} phase for layer_instance {} layer_name {} layer_type {} ".format(
                l_phase, layer_instance, l_name, l_type))

        return [l_type, l_phase, l_name]

    ###################################################
    # begin mxnet
    ###################################################
    def mxnet_check_add_dgrad(self, layer_name, report_tbl):
        """
        Checks the name of the layer that follows an add_grad
        kernel, and assigns that layer name to the add_grad kernel.
        This function should only work on the layer immediately
        following the add_grad.
        Args:
            layer_name (str): The name of the current layer
            report_table (dict): The dictionary to update
        """
        # Attatch add_dgrad to the correct layer
        if self.prev_layer_name is not None and layer_name is not None:
            assert re.search(r"sum_grad", self.prev_layer_name),\
                    "Unexpected prev layer name {}".format(self.prev_layer_name)
            report_tbl['LayerName'][-1] = self.prev_layer_name + '_' + layer_name
            # So we can reuse this check
            self.prev_layer_name = None



    def mxnet_update_fc_fwd_kernel_dict(self, layer_name, forward_kernel_name):
        """Updates the dict of fully connected forward GEMM kernel names.
        This makes it possible to identify whether bprop kernels are
        wgrad or dgrad (see mxnet_assign_fc_dgrad_wgrad())
        """
        k = forward_kernel_name.lower()

        # There are sometimes multiple kernels in a forward layer. We want
        # the GEMM kernel.
        if k.find('gemm') >= 0 or k.find('mma') >= 0 or k.find('s884') >= 0:
            self.mxnet_fwd_layer_name_dict.update({ layer_name:forward_kernel_name })

    def mxnet_assign_fc_dgrad_wgrad(self, layer_name, backward_kernel_name):
        """Assigns phase to fully connected backprop layers based 
        on fully connected fprop layer input shapes. In the forward pass 
        GEMM, matrix A is the weight tensor and matrix B is the data tensor.
        In the backward pass, matrix A is the data tensor and B is the 
        (weight or data) tensor we take the gradient with respect to.
        
        'tn' means that in the GEMM, matrix A is transposed and B is normal.
        'nt' means that in the GEMM, matrix A is normal and B is transposed.
        From this information in the forward and backward GEMM kernel names,
        we deduce whether the backward GEMM is wgrad or dgrad. 
        
        Caveats: 
            - This function only works when exactly one of the forward tensors 
            is transposed.
            - This function depends on calling the forward pass before the 
            backward pass.
            - This function uses a global dict of forward kernel names, which 
            is not the best way to do this. At some point it would be good to 
            refactor this into a class.

        Args:
            layer_name (str): the unique layer name corresponding to this kernel
            kernel_name (str): the kernel name
        Returns:
            phase (str): the phase corresponding to this kernel
            confidence (bool): whether this is a confident assumption.
        """
        phase = 'dgrad'
        phase_confidence = 1
        backward_kernel_name = backward_kernel_name.lower()
        # Gradient with respect to biases is wgrad
        if backward_kernel_name.find('addbiasgrad') >= 0:
            phase = 'wgrad'
        # Parse the corresponding forward kernel name to find out 
        # whether the GEMM is dgrad or wgrad.
        elif layer_name in self.mxnet_fwd_layer_name_dict:
            forward_kernel_name = self.mxnet_fwd_layer_name_dict[layer_name]
            if forward_kernel_name.find('_tn') >= 0:
                if backward_kernel_name.find('_nt') >= 0:
                    phase = 'wgrad'
                elif backward_kernel_name.find('_nn') >= 0:
                    phase = 'dgrad'
            elif forward_kernel_name.find('_nt') >= 0:
                if backward_kernel_name.find('_nn') >= 0:
                    phase = 'wgrad'
                elif backward_kernel_name.find('_nt') >= 0:
                    phase = 'dgrad'
            # Assign the first GEMM to be wgrad, second to be dgrad
            else:
                phase_confidence = 0
                # If we have not yet processed a backward GEMM for this layer, assign wgrad
                if layer_name not in self.mxnet_bwd_layer_name_dict:
                    self.mxnet_bwd_layer_name_dict[layer_name] = backward_kernel_name
                    phase = 'wgrad'
                else:
                    phase = 'dgrad'

        return phase, phase_confidence

    def get_mxnet_layer_type_from_name(self, layer_instance, kernel_name):
        """Processes a row in the MARKER table from nvprof to extract the
        layer type (i.e. Convolution, BatchNorm, etc.) and layer name (i.e.
        stage3_unit5_conv3, etc.). Also extracts some phase information from
        the row, and infers missing phase information from the kernel name.
        """
        phase_confidence = 1
        dims = ""
        kernel_name = kernel_name.lower()

        ## Save all the nvtx strings in order of appearance
        ## Then emit them to a file for post processing
        self.saved_nvtx_strings.append(layer_instance)

        layer_instance = layer_instance.strip('[')  
        layer_instance = layer_instance.rstrip(']') 
        if self.mxnet_print_interval==0:
            print(layer_instance)
        # MXNET NVTX markers contain the TYPE followed by the NAME
        # The syntax depends if it is using the symbolic or Gluon API
        # Convolution:name=ssd0_resnetmlperf0_stage1_conv0_fwd (Gluon)
        # Convolution{name=conv0 (Symbolic)
        try:
            ins = layer_instance.split(";")
            for i in ins:
                if i.find("in0")>=0:
                    dims = i
        except:
            pass
        classic_layer_regex = re.compile(r"(?P<type>[a-zA-Z0-9_]+)({|:)name=(?P<name>[a-zA-Z0-9_]+)")
        classic_match = re.match(classic_layer_regex,layer_instance)
        
        # In the Gluon API, some layers can look like this
        # ImperativeBulk: Convolution
        # ImperativeBulk: _backward_mean
        # weight_update operations always begin with ImperativeBulk, for example
        # ImperativeBulk: multi_mp_sgd_mom_update
        # unfortunately, in this syntax, there is no name to extract
        imperative_layer_regex = re.compile(r"ImperativeBulk: (?P<type>[a-zA-Z0-9_]+)")
        imperative_match = re.match(imperative_layer_regex,layer_instance)
        
        def ltype_to_dlsim_type(ltype):
            if re.match(r"((elemwise_(add|mul|div|sub))|(_*(mul|minus|greater|maximum)_scalar)|(sum|clip|abs|mean|div|mul|sub))", ltype):
                #import pdb; pdb.set_trace()
                return "Eltwise"
            else:
                return ltype
                
        def check_if_wgrad(layer_instance, kernel_name):
            lphase = 'dgrad'
            if layer_instance.find('wgrad') >= 0 or kernel_name.find('wgrad') >= 0:
                lphase = 'wgrad'
            return lphase

        if classic_match:
            lname = classic_match.group('name')
            lname = lname.replace("_backward","")
            ltype = classic_match.group('type')
            if ltype.find("_backward_") >= 0:
                ltype = ltype.replace("_backward_","")
                lphase = check_if_wgrad(layer_instance, kernel_name)
                if ltype.find("FullyConnected") >= 0:
                    # This is the best check we have for dgrad vs. wgrad - works on
                    # resnet-v1b-fl, mileage may vary with other networks
                    lphase, phase_confidence = self.mxnet_assign_fc_dgrad_wgrad(lname, kernel_name)
                else:
                    ltype = ltype_to_dlsim_type(ltype)
            else:
                lphase = 'fprop'
                ## This is the join operation of 2 dgrad
                ## diff tensors that need to be added before
                ## the total gradient flows into the next op
                ## They flow into a fwd pass node that resulted in a split
                if lname.find("sum_grad") >= 0:
                    lphase = "dgrad"
                    ltype = "Eltwise"
                    self.prev_layer_name = lname
                    lname = None
                else:
                    if ltype.find("FullyConnected") >= 0:
                        self.mxnet_update_fc_fwd_kernel_dict(lname, kernel_name)
                    else:
                        ltype = ltype_to_dlsim_type(ltype)
            if self.mxnet_print_interval==0:
                self.mxnet_print_interval = 1000
                #print(ltype,lphase,lname,ltype)
            self.mxnet_phase = lphase
            self.mxnet_print_interval = self.mxnet_print_interval - 1
        elif imperative_match:
            ltype = imperative_match.group('type')
            if ltype.find("_backward_") >= 0:
                ltype = ltype.replace("_backward_","")
                lname = ltype
                ltype = ltype_to_dlsim_type(ltype)
                lphase = check_if_wgrad(layer_instance, kernel_name)
            elif ltype.find("sgd_mom_update") >= 0:
                lname = ltype
                ltype = "weight_update"
                lphase = "weight_update"
            else:
                lphase = 'fprop'
                lname = ltype
                ltype = ltype_to_dlsim_type(ltype)
        else:
            lphase = self.mxnet_phase            
            lname  = layer_instance.strip('_')  
            ## Argmax layers don't have the same NVTX signature as the rest
            if re.search(r"argmax", lname):
                ltype = "Argmax"
                lphase = "dgrad"
            else:
                ltype = ltype_to_dlsim_type(lname)
        return (ltype, lphase, lname, ltype, phase_confidence, dims)

    def decode_nvtx_marker_layer_string(self, layer_instance, kernel_name=None, layersjson=None,\
                                        marker_stack=[]):
        """
        Each framework has a different way of encoding the layer info in the NVTX marker string.
        So far Caffe2, KNF, TensorFlow, are supported
        Also - there is format for decoding the layer for cases where we control the NVTX string in the workload.
        """
        # Default confidence for phase assignment is 1. Should be 0 for kernels where we guess the phase (see MXNet)
        phase_confidence = 1
        dims = ""
        l_op_name = None
        l_tensors = None
        if self.FwType == "FW_TENSORRT":
            from netex.network_exporters.nvprof2json import get_json_from_instance
            from netex.network_exporters.nvprof2json_old import get_json_from_instance_old
            if self.export_trt_json:
                layer_instance = layer_instance.replace('\n\t', ', ')
                layer_instance = layer_instance.replace('\n', '')
                layer_instance = layer_instance.replace('(not clipped for now) ', '')
                ldesp = re.match(r'.+\s[(](Name:\s.+)[)]$', layer_instance)
                #print("{}".format(layer_instance))
                onejson = None
                if ldesp:
                    onejson = get_json_from_instance(ldesp.group(1))
                else:
                    ldesp = re.match(r'(.+) [(](CUDA|CASK)( .+)[)]$', layer_instance)
                    if ldesp:
                        onejson = get_json_from_instance_old(ldesp.group(0))
                if onejson and onejson not in layersjson['Layers']:
                    layersjson['Layers'].append(onejson)
                    l_type, l_phase, l_name = onejson['LayerType'], 'Fprop', convert_uniq_lname(onejson['Name'])
                else:
                    l_type, l_phase, l_name = self.get_tensorrt_layer_type_from_name(layer_instance)
            else:
                l_type, l_phase, l_name = self.get_tensorrt_layer_type_from_name(layer_instance)
        elif self.FwType == "FW_PYTORCH":
            l_type, l_phase, l_name, l_tensors =\
                    self.get_pyt_exporter_layer_type_from_name(layer_instance, marker_stack)
            #l_type, l_phase, l_name = get_pytorch_layer_type_from_name(layer_instance)
        elif self.FwType == "FW_TENSOR_FLOW":
            if self.graph_input_file:
                l_type, l_phase, l_name, l_op_name =\
                    self.get_tensorflow_layer_info_with_map(layer_instance, kernel_name)
            else:
                l_type, l_phase, l_name = self.get_tensorflow_layer_type_from_name(layer_instance)
        elif self.FwType=="FW_MXNET":
            l_type, l_phase, l_name, l_op_name, phase_confidence, l_tensors = self.get_mxnet_layer_type_from_name(layer_instance, kernel_name)
        elif self.FwType == 'FW_CAFFE2':
            res = re.search(r"(\w+)\s+\((\S+)\)", layer_instance)
            operator = res.group(1)
            layer_instance = res.group(2)
            ## Reset layer_type, phase, layer_name for caffe2
            ## layer_name return value may be redundant
            l_type, l_phase, l_name = self.get_caffe2_layer_type_from_name(operator, layer_instance)
        else:
            l_type, l_phase, l_name = self.get_layer_type_from_name(layer_instance)

        return [l_type, l_phase, l_name, l_op_name, phase_confidence, l_tensors]

    def scan_pyt_marker(self, marker):
        fields = re.split(', sizes', marker)
        assert len(fields) == 2, "Unexpected Marker string {}".format(marker)
        marker_text = fields[0]
        tensor_sizes = eval(re.sub('\s*=\s+', '', fields[1]))
        return marker_text, tensor_sizes

    def get_pyt_exporter_layer_type_from_name(self, marker_str, marker_stack):
        '''
        get_pyt_exporter_layer_type_from_name()

            Pytorch NVTX markers use push/pop so there is often a stack of markers around a single
            kernel.  In order to link the kernel name to the layer name you need to include the
            whole stack  because only one of these markers has the layer information in it.
        '''
        ## Assumes that the profiler was configured to include tensor sizes
        l_type   = 'NA'
        l_name   = 'NA'
        op_info  = ''
        tensor_sizes = []
        print("Marker -> {}".format(marker_str))
        for marker in marker_stack:
            print("Sub marker -> {}".format(marker))
            marker_text, tensor_sizes = self.scan_pyt_marker(marker)
            if marker_text:
                op_info = marker_text
                break
        if op_info == '':
            return (l_type, self.phase, l_name, tensor_sizes)
        pat = re.match('(\S+),\s+seq\s+=\s+(\d+)', op_info)
        assert pat, "OpName, seq pattern match faild for {}".format(op_info)
        if pat:
            op_name = pat.group(1)
            seq_id  = int(pat.group(2))
            if self.min_seq_id < 0:
                self.min_seq_id = seq_id

        if self.phase == 'Fprop':
            if re.search('Backward', op_name):
                self.phase = 'Dgrad'

        print("Op Name {} seq id {} tensors {} ".format(op_name, seq_id, tensor_sizes))
        l_type = op_name
        l_name = "{}".format(seq_id)
        return l_type, self.phase, l_name, tensor_sizes

    def get_tensorrt_layer_type_from_name(self, name=None):
        """Parse the long form layer name for tensorRT."""
        layer_type = 'UNK_LAYER_TYPE'
        layer_name = "UNK_LAYER_NAME"
        Phase = 'Fprop'
        if name is None:
            print("Error get_layer_type_from_name - Bad args - exiting...")
            sys.exit(1)
        try:
            layerinjson = json.loads(name)
        except ValueError:
            return layer_type, Phase, name
        if layerinjson:
            if 'Name' in layerinjson:
                ## Strip off " [profile *]" from the layer name
                ## (layer name with/without [profile *] should be the same from DLSim's perspective)
                layer_name = uniq_layer_name(layerinjson['Name'])
            if 'ParameterType' in layerinjson:
                layer_type = layerinjson['ParameterType']
            elif 'LayerType' in layerinjson:
                layer_type = layerinjson['LayerType']
        if (layer_type == "UNK_LAYER_TYPE" or layer_name == "UNK_LAYER_NAME"):
            print("-W- Couldn't deciper Layer Name or Layer Type from {} ".format(name))
        return layer_type, Phase, layer_name

    def get_layer_type_from_name(self, name=None):
        """Get the layer type from the long form layer name."""
        if name is None:
            print("Error get_layer_type_from_name - Bad args - exiting...")
            sys.exit(1)
        #print(name)
        phase, layer_name = name.split(' ')
        layer_type = layer_name

        ## @@@ For new NVTX - make the convension 'Phase LayerType,UniqueLayerName'
        pattern = re.compile(r"([a-zA-Z0-9]+),(\S+)")
        res = re.match(pattern, layer_name)
        if res is not None:
            layer_type = "{}".format(res.group(1))
            layer_name = "{}".format(res.group(2))
            return layer_type, phase, layer_name
        '''
        ## @@@ For Deep Bench - Remove this - make Deep Bench follow 'Phase Type,UniqueName' pattern
        pattern = re.compile(r"(Conv_\d+x\d+)")
        res = re.match(pattern, layer_name)
        if res is not None:
            layer_type = "{}".format(res.group(1))
            return layer_type, phase, layer_name
        '''

        ### All remaining pattern matches are there to support KNF naming convention

        pattern = re.compile(r"layer_\d+_\d+_(\w+)")
        res = re.match(pattern, layer_name)
        if res is not None:
            layer_type = "{}".format(res.group(1))
            return layer_type, phase, layer_name

        ## Look for res_branch_relu tag
        #pattern = re.compile(r"res\w+_branch\w+_(relu)")
        pattern = re.compile(r"res\w+[_]+(relu)")
        res = re.match(pattern, layer_name)
        if res is not None:
            layer_type = "{}".format(res.group(1))
            return layer_type, phase, layer_name

        ## Look for res_branch tag
        pattern = re.compile(r"res\w+_branch\w+")
        res = re.match(pattern, layer_type)
        if res is not None:
            layer_type = "conv"
            return layer_type, phase, layer_name

        ## Look for bn_branch tag
        pattern = re.compile(r"(bn)\w+_branch\w+")
        res = re.match(pattern, layer_type)
        if res is not None:
            layer_type = "{}".format(res.group(1))
            return layer_type, phase, layer_name

        pattern = re.compile(r"res\d+[a-f]")
        res = re.match(pattern, layer_type)
        if res is not None:
            if self.Debug:
                print("Found elt layer type from {}".format(layer_type))
            layer_type = "elt"
            return layer_type, phase, layer_name

        # Get rid of numbers
        layer_type = re.sub(r"\d+", "", layer_type)

        # Special case - conv_expand - is a conv layer
        pattern = re.compile(r"(\w+)_expand")
        res = re.match(pattern, layer_type)
        if res is not None:
            layer_type = "{}".format(res.group(1))
            return layer_type, phase, layer_name

        ## Look for bn_conv - V1 prototxt format has bn as first field V2 has it as 2nd field
        pattern = re.compile(r"bn_(conv)")
        res = re.match(pattern, layer_type)
        if res is not None:
            layer_type = "bn"
            return layer_type, phase, layer_name

        ## Look for compound layer names - use the 2nd field for the name
        layer_type = re.sub(r".*_(\w+)", "\g<1>", layer_type)

        return layer_type, phase, layer_name

    def demangle_kernel_name(self, mangled_name=None):
        """Kernel names are mangled use c++filt to get human readable names."""

        if mangled_name not in self.kernel_hash:
            ret = subprocess.run(['c++filt', mangled_name], stdout=subprocess.PIPE)
            new_name = ret.stdout.decode("utf-8").strip()
            self.kernel_hash[mangled_name] = new_name
            if self.Debug:
                print("C++filt kernel name -> {}".format(new_name))
        else:
            new_name = self.kernel_hash[mangled_name]

        return new_name

    def get_tbl_name_from_type(self, tbl_type=None, tbl_list=None):
        """
        Return full table name that matches the tbl_type substring
        """
        tbl_name = None
        if tbl_type is None:
            print("Error get_tbl_name_from_pattern: No tbl_type specified - exiting.\n")
            sys.exit(1)

        if tbl_list is None:
            print("Error get_tbl_name_from_pattern: No tbl_list specified - exiting.\n")
            sys.exit(1)

        ## Walk the list of tbls - return the one that has substring tbl_type
        for tbl in tbl_list:
            pattern = re.compile(tbl_type)
            if pattern.search(tbl):
                tbl_name = tbl
                break

        return tbl_name

    def get_tbl_marker_by_time_window(self, cpu_start=None, cpu_end=None, thread_id=None, pd_frame=None, tbl_size=None):
        """
        Find the marker / range whose start and end times cover the cpu event start and end times passed in.
        @@@ This function is slow - sql look ups seem to take really long
        Try using Pandas instead - create a frame
        @@@ Instead of doing a query every time this is called - should read
        in the table once - then keep a pointer to the last row that was selected
        start the next search from the row pointer - this should save a lot of time
        because each search will only be a couple iterations
        Search by time stamp
        return the fields
        """

        if cpu_start is None or cpu_end is None or pd_frame is None or thread_id is None:
            print("get_tbl_marker_by_time_window: Bad args - exiting ")
            sys.exit(1)

        marker_end = None
        pd_marker_id = None
        '''
        ## Initialize index first time it sees thread_id
        if thread_id not in marker_tbl_index:
            marker_tbl_index[thread_id] = 0

        row_index = marker_tbl_index[thread_id]
        for row in range(row_index, marker_tbl_size):
            marker_tbl_index[thread_id] += 1
            frame_row = pd_frame.iloc[row]
            if frame_row['timestamp'] > cpu_end and frame_row['thread_id'] == thread_id:
                pd_marker_id = frame_row['id']
                marker_end   = frame_row['timestamp']
                break

        if (marker_end is None or pd_marker_id is None) and (row_index < marker_tbl_size):
            raise Exception("Query failed for timestamp  > {} and thread_id == {} row index {} timestamp (ms) {}".format(cpu_end, thread_id, row_index, (cpu_end-time_base)/1000000)) 
        '''

        ## Get the first entry whose time stamp is > end
        ## record the ID - then do a 2nd query that returns name when ID == id from prev query
        #query_string     = "(timestamp > {}) & (thread_id == {})".format(cpu_end, thread_id)
        #tmp_frame        = pd_frame.query(query_string)
        tmp_frame = pd_frame[(pd_frame['end_time'] > cpu_end) & (pd_frame['thread_id'] == thread_id)]
        if tmp_frame.empty:
            raise Exception("Query failed for timestamp  > {} and thread_id == {}".format(cpu_end, thread_id))
        pd_marker_id = tmp_frame['id'].iat[0]
        marker_end = tmp_frame['end_time'].iat[0]
        marker_start = tmp_frame['start_time'].iat[0]
        pd_name_id = tmp_frame['name_id'].iat[0]
        marker_name = self.string_hash[pd_name_id]

        ## 2nd Query using ID to get name and marker start time
        #query_string     = "id == {}".format(pd_marker_id)
        #tmp_frame        = pd_frame.query(query_string)
        #pd_name_id       = tmp_frame['name_id'].iat[0]
        #marker_name         = string_hash[pd_name_id]
        #marker_name      = tmp_frame['name'].iat[0]
        #marker_start     = tmp_frame['timestamp'].iat[0]
        ## Reset the marker tbl index to the start time that maps to this id
        #marker_tbl_index[thread_id] = tmp_frame.index[0]

        if (self.Debug):
            print("Marker name {} start {} end {} ".format(marker_name, marker_start, marker_end))
        return marker_name, marker_start, marker_end

    def process_runtime_tbl(self, tbl=None, cur=None):
        if tbl is None:
            print("Error process_runtime_tbl: No tbl specified - exiting.\n")
            sys.exit(1)
        if cur is None:
            print("Error process_runtime_tbl: No cursor specified - exiting.\n")
            sys.exit(1)

        pattern = re.compile('RUNTIME')
        if pattern.search(tbl):
            cmd_string = "select * from {};".format(tbl)
            print("Executing sql cmd {}".format(cmd_string))
            cur.execute(cmd_string)  ## Need to use a tuple for variable sub- even though only passing 1 value
            tbl_hdr = get_tbl_hdrs(cur, self.Debug)
            self.dump_rows(cur, tbl_hdr, 'RUNTIME')
        return

    def decode_trt_kernel_name(self, nvtx_json):
        if self.FwType == "FW_TENSORRT" and nvtx_json:
            try:    
                layer_info = json.loads(nvtx_json)
            except (ValueError, TypeError):
                return False
            if layer_info and "TacticName" in layer_info:
                return layer_info["TacticName"]
        return False

    def load_input_yml_files(self):
        if nsys_parser.graph_input_file:
            with open(nsys_parser.graph_input_file, "r") as yml_fd:
                nsys_parser.graph_info_map = load(yml_fd, Loader=Loader)

        if nsys_parser.weight_to_fprop_file:
            with open(nsys_parser.weight_to_fprop_file, "r") as yml_fd:
                nsys_parser.weight_to_fprop_map = load(yml_fd, Loader=Loader)

        if nsys_parser.xla_kernel_file:
            with open(nsys_parser.xla_kernel_file, "r") as yml_fd:
                nsys_parser.xla_kernel_map = load(yml_fd, Loader=Loader)
        return

################################################################################
## Main program
################################################################################
if __name__ == "__main__":

    nsys_parser = NsysParser()
    nsys_parser.parse_cmd_line()
    #excel_writer = pd.ExcelWriter(nsys_parser.excel_file_name, engine='xlsxwriter')
    output_fd = nsys_parser.open_ouput_file()
    
    nsys_parser.load_input_yml_files()

    ## Make 2 new sheets - 1 has the raw data 1 has average values
    frame_list = []
    ave_val_frame_list = []
    for db_file in nsys_parser.db_file_list:
        pd_frame = nsys_parser.read_db_file(db_file, output_fd)
        frame_list.append(pd_frame)
        if nsys_parser.ComputeAverage:
            print("Computing average runtime over number of iterations")
            start = time.time()
            pd_ave_val_frame = nsys_parser.compute_ave_runtime(pd_frame)
            end = time.time()
            print('Average runtime compute time: {} sec'.format(float((end - start) / (60 * 60))))
            ave_val_frame_list.append(pd_ave_val_frame)

        ## Make 1 worksheet per experiment
        panda_sheet = os.path.basename(db_file)
        # Drop the file extension
        (panda_prefix, ext) = os.path.splitext(panda_sheet)
        panda_sheet = panda_prefix + '.csv'
        panda_sheet = os.path.join(nsys_parser.result_dir, panda_sheet)
        print("Writing CSV file {} ".format(panda_sheet))
        pd_frame.to_csv(panda_sheet, sep=',')

    ## Make a combined pivot table + a 2nd sheet that takes average over all iterations
    panda_sheet += 'combined_tbl'
    panda_ave_val_sheet = panda_sheet + "_ave.csv"
    panda_sheet = panda_sheet + '.csv'
    panda_sheet = os.path.join(nsys_parser.result_dir, panda_sheet)
    panda_ave_val_sheet = os.path.join(nsys_parser.result_dir, panda_ave_val_sheet)
    pivot_tbl_frame = pd.concat(frame_list)
    ## Combine all the frames into 1
    print("Writing Xcel worksheet {} ".format(panda_sheet))
    pivot_tbl_frame.to_csv(panda_sheet, sep=',')

    if nsys_parser.ComputeAverage:
        print("Writing Xcel worksheet {} ".format(panda_sheet))
        ave_val_pivot_tbl_frame = pd.concat(ave_val_frame_list)
        ave_val_pivot_tbl_frame.to_csv(panda_ave_val_sheet, sep=',')
        ave_val_pivot_tbl_frame.to_excel(os.path.join(nsys_parser.result_dir, panda_sheet+"_ave.xlsx"))

    ## Close the xcel sheet
    #excel_writer.save()
    if nsys_parser.pivot_tbl is not None:
        output_fd.close()

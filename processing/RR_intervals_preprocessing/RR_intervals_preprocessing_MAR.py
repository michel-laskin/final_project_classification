from tool_box_MAR import *



#-----------------------------------------------------------------------------------------------------------
# Extract RR intervals :
def extract_rr(signal,params):
    peaks,_=find_peaks(signal, distance=params["d"], prominence=params["p"])
    rr_locations=np.diff(peaks)
    rr=rr_locations*(1/(params["hz"])*1000)
    return rr


#-----------------------------------------------------------------------------------------------------------
#Range-based filtering (RBF)

def RBF(RR_Intervals,params):
    if params["apply_rbf"]:
        low_limit=60000/params["max_HR"]
        high_limit=60000/params["min_HR"]
        RR_Intervals=[RR for RR in RR_Intervals if low_limit<=RR<=high_limit]         
    return RR_Intervals


#-----------------------------------------------------------------------------------------------------------
# Moving average filter (MAF):

def MAF(RR_Intervals,params):
    if not params["apply_maf"]:
        return RR_Intervals
    
    k = params["window_size_maf"] // 2  
    filtered_RR = []
    filtered_RR.extend(RR_Intervals[:k])

    for i in range(k, len(RR_Intervals) - k):
        neighbors = np.concatenate([RR_Intervals[i-k : i], RR_Intervals[i+1 : i+k+1]])
        avg = np.mean(neighbors)
        lower_bound = avg * (1 - params["limit_value_maf"]/100)
        upper_bound = avg * (1 + params["limit_value_maf"]/100)
        
        if lower_bound <= RR_Intervals[i] <= upper_bound:
            filtered_RR.append(RR_Intervals[i])
        else:
            pass
    filtered_RR.extend(RR_Intervals[-k:])
    
    return filtered_RR

#-----------------------------------------------------------------------------------------------------------
# Quotient filter (QF):

def QF(RR_Intervals,params):
    if not params["apply_qf"]:
        return RR_Intervals
    
    filtered_RR = []
    filtered_RR.append(RR_Intervals[0])
    for i in range(1,len(RR_Intervals)-1):
        lower_bound=1-params["limit_value_qf"]/100
        upper_bound=1+params["limit_value_qf"]/100
        ratio1=RR_Intervals[i]/RR_Intervals[i-1]
        ratio2=RR_Intervals[i-1]/RR_Intervals[i]
        ratio3=RR_Intervals[i]/RR_Intervals[i+1]
        ratio4=RR_Intervals[i+1]/RR_Intervals[i]


        if lower_bound<=ratio1<=upper_bound and lower_bound<=ratio2<=upper_bound and lower_bound<=ratio3<=upper_bound and lower_bound<=ratio4<=upper_bound:
            filtered_RR.append(RR_Intervals[i])
        else:
            pass
    filtered_RR.append(RR_Intervals[-1])    
    return filtered_RR



import numpy as np
# from Chandra.Time import DateTime
from cxotime import CxoTime
from kadi import events
# from Ska.engarchive import fetch_eng as fetch
from cheta import fetch_eng as fetch

healthcheck_msids = ['HRMA_AVE','4OAVHRMT','HRMHCHK','OBA_AVE','4OAVOBAT','TSSMAX','TSSMIN',
                     'TABMAX','TABMIN','THSMAX','THSMIN','OHRMGRD3','OHRMGRD6']

dwell_mode_msids = ['4PRT5BT', '4RT585T', 'AFLCA3BI', 'AIRU1BT', 'CSITB5V', 'CUSOAOVN', 'ECNV3V',
                    'PLAED4ET', 'PR1TV01T', 'TCYLFMZM', 'TOXTSUPN', 'ES1P5CV', 'ES2P5CV']

def filter_safing_actions(times, stat, safe_modes=True, nsm_modes=True, pad=None, transitions_only=False, tpad=None):


    starts = []
    stops = []
    if safe_modes == True:
        starts.extend(list(events.safe_suns.table['tstart']))
        stops.extend(list(events.safe_suns.table['tstop']))

    if nsm_modes == True:
        starts.extend(list(events.normal_suns.table['tstart']))
        stops.extend(list(events.normal_suns.table['tstop']))

    intervals = list(zip(starts, stops))

    if pad:
        intervals = [(i1 - pad[0], i2 + pad[1]) for i1, i2 in intervals]

    if transitions_only:
        if not tpad:
            tpad = (328, 328, 328, 328)

        intervals_start = [(i1 - tpad[0], i1 + tpad[1]) for i1, i2 in intervals]
        intervals_stop = [(i2 - tpad[2], i2 + tpad[3]) for i1, i2 in intervals]
        intervals = intervals_start + intervals_stop

    stat = str(stat).lower()

    # import code
    # code.interact(local=dict(globals(), **locals())) 

        
    keep = filter_intervals(times, intervals, stat)

    return keep


def find_span_indices(dval):
    dval = list(dval)
    dval.insert(0, False) # Prepend to make indices line up
    idata = np.array(dval, dtype=type(1))
    d = np.diff(idata)

    starts = d == 1
    stops = d == -1

    starts = list(starts)
    stops = list(stops)

    if idata[-1] == 1:
        stops.insert(-1, True)

    starts = np.where(starts)[0]
    stops = np.where(stops)[0]
    
    return list(zip(starts, stops))


def get_dwell_mode_intervals(t1='2015:001:00:00:00', t2=None):
    if t2 is None:
        t2 = CxoTime().date
    data = fetch.Msid('ctudwlmd', t1, t2, stat=None)
    bad = data.raw_vals == 1

    if any(bad):
        spans = find_span_indices(bad)
        timespans_en = [(CxoTime(data.times[ind[0]]).date, CxoTime(data.times[ind[1]]).date) for ind in spans]
        timespans_en = [(CxoTime(t1).secs - 33, CxoTime(t2).secs + 33) for t1, t2 in timespans_en]
        timespans_en = [(CxoTime(a).date, CxoTime(b).date) for a, b in timespans_en]
        return timespans_en

    else:
        return []


def filter_outliers(vals, mad_const=None):

    if mad_const is None:

        std = np.std(vals)
        keep = ~(DoubleMADsFromMedian(vals) > 0.1 * std)

        const = 0.1 * std

        if len(keep[keep == True]) < .95 * len(keep):
            keep = ~(DoubleMADsFromMedian(vals) > 0.5 * std)

        if len(keep[keep == True]) < .95 * len(keep):
            keep = ~(DoubleMADsFromMedian(vals) > std)

        if len(keep[keep == True]) < .95 * len(keep):
            keep = ~(DoubleMADsFromMedian(vals) > 5000 * std)

        if len(keep[keep == True]) < .95 * len(keep):
            keep = ~(DoubleMADsFromMedian(vals) > 50000 * std)

    else:
        keep = ~(DoubleMADsFromMedian(vals) > mad_const)

    return keep


def clean_ltt_data(telem, stat, mad_const=None, outlier_filter_fields=('vals')):

    keep = filter_safing_actions(telem.times, stat, safe_modes=True, nsm_modes=True, transitions_only=True)

    for name in telem.colnames:
        telem.__dict__[name] = telem.__dict__[name][keep]

    keeps = []
    if (mad_const is not None) and (mad_const > 0):
        for field in outlier_filter_fields:
            keeps.append(filter_outliers(telem.__dict__[field], mad_const=mad_const))

    merged_keep = np.array([True] * len(telem.times))
    for keep in keeps:
        merged_keep = merged_keep & keep

    return merged_keep


def DoubleMAD(x, zeromadaction="warn"):
    """ Core Median Absolute Deviation Calculation    

    :param x: 1 dimenional data array
    :param zeromadaction: determines the action in the event of an MAD of zero, anything other than 'warn' will throw an exception

    """

    m = np.median(x)
    absdev = np.abs(x - m)
    leftmad = np.median(np.abs(x[x<=m]))
    rightmad = np.median(np.abs(x[x>=m]))
    if (leftmad == 0) or (rightmad == 0):
        if zeromadaction.lower() == 'warn':
            print('Median absolute deviation is zero, this may cause problems.')
        else:
            raise ValueError('Median absolute deviation is zero, this may cause problems.')
    return leftmad, rightmad


def DoubleMADsFromMedian(x, zeromadaction="warn"):
    """ Median Absolute Deviation Calculation    

    :param x: 1 dimenional data array
    :param zeromadaction: determines the action in the event of an MAD of zero, anything other than 'warn' will throw an exception

    """
    
    def DoubleMAD(x, zeromadaction="warn"):
        """ Core Median Absolute Deviation Calculation    
        """

        m = np.median(x)
        absdev = np.abs(x - m)
        leftmad = np.median(np.abs(x[x<=m]))
        rightmad = np.median(np.abs(x[x>=m]))
        if (leftmad == 0) or (rightmad == 0):
            if zeromadaction.lower() == 'warn':
                print('Median absolute deviation is zero, this may cause problems.')
            else:
                raise ValueError('Median absolute deviation is zero, this may cause problems.')
        return leftmad, rightmad
    
    twosidedmad = DoubleMAD(x, zeromadaction)
    m = np.median(x)
    xmad = np.ones(len(x)) * twosidedmad[0]
    xmad[x > m] = twosidedmad[1]
    maddistance = np.abs(x - m) / xmad
    maddistance[x==m] = 0
    return maddistance



def get_intervals(msid, setname=''):
    setname = setname.lower()
    msid = msid.lower()

    individual_msid_intervals = {'3fapsat': [ ('2011:190:19:43:58.729', '2011:190:19:44:58.729'),
                                              ('2012:151:10:42:03.272', '2012:151:10:43:03.272'),
                                              ('2015:264:20:48:00.000', '2015:264:20:51:00.000'),
                                              ('2016:064:00:53:00.000', '2016:064:00:56:00.000'),
                                              ('2018:285:12:49:00.000', '2018:285:12:53:00.000'),
                                              ('2020:145:14:16:00.000', '2020:145:14:20:00.000'),
                                              ('2020:146:16:25:00.000', '2020:146:16:45:00.000')],
                                 '3tspzdet':  [('2018:304:01:30:00.000', '2018:304:02:00:00.000')],
                                 'ohrthr15':  [('2004:187:15:38:05.785', '2004:187:15:39:05.785')],
                                 'ohrthr21':  [('2011:187:12:28:11.387', '2011:187:12:29:11.387')],
                                 'ohrthr22':  [('2011:187:12:28:11.387', '2011:187:12:29:11.387')],
                                 'pline10t':  [('2003:127:03:50:14.592', '2003:127:03:51:14.592')],
                                 'pline16t':  [('2000:318:13:02:33.845', '2000:318:13:03:33.845')],
                                 'tasprwc':   [('2002:296:01:42:08.282', '2002:296:01:43:08.282')],
                                 'tcylaft2':  [('2001:005:08:25:32.419', '2001:005:08:26:32.419')],
                                 'tcyz_rw6':  [('2000:302:18:58:37.793', '2000:302:18:59:37.793')],
                                 'tep_bpan':  [('2003:008:20:09:32.562', '2003:008:20:10:32.562'),
                                               ('2016:064:01:09:00.000', '2016:064:01:11:00.000')],
                                 'tep_pcu':   [('2016:064:01:09:00.000', '2016:064:01:11:00.000')],
                                 'tep_eia':   [('2016:064:01:09:00.000', '2016:064:01:11:00.000')],
                                 'tep_rctu':  [('2016:064:01:09:00.000', '2016:064:01:11:00.000')],
                                 'tep_psu1':  [('2016:064:01:09:00.000', '2016:064:01:11:00.000')],
                                 'tep_psu2':  [('2016:064:01:09:00.000', '2016:064:01:11:00.000')],
                                 'tep_ppan':  [('2016:064:01:09:00.000', '2016:064:01:11:00.000')],
                                 'tfspcmp':   [('2011:299:04:57:52.020', '2011:299:04:58:52.020')],
                                 'tmysada':   [('2011:190:19:47:48.329', '2011:190:19:48:48.329')],
                                 'tmzp_cnt':  [('2000:299:16:20:20.183', '2000:299:16:21:20.183')],
                                 'trspotep':  [('2003:127:03:50:14.592', '2003:127:03:51:14.592')],
                                 'tsctsf2':   [('2000:055:15:23:18.602', '2000:055:15:24:18.602')],
                                 'tsctsf4':   [('2020:145:14:17:00.000', '2020:145:14:18:00.000'),
                                               ('2020:146:16:28:00.000', '2020:146:16:30:00.000')],
                                 'cxpnait':   [('2011:299:04:57:00.000', '2011:299:05:00:00.000'),
                                               ('2016:064:00:52:00.000', '2016:064:00:56:00.000')],
                                 'aacccdpt':  [('2011:187:12:00:00.000', '2011:191:00:00:00.000'),
                                               ('2011:299:04:00:00.000', '2011:300:00:00:00.000'),
                                               ('2012:150:03:00:00.000', '2012:151:18:00:00.000'),
                                               ('2015:252:13:00:00.000', '2015:252:15:00:00.000'),
                                               ('2015:264:04:00:00.000', '2015:265:00:00:00.000'),
                                               ('2015:252:12:00:00.000', '2015:253:00:00:00.000'),
                                               ('2015:264:00:00:00.000', '2015:267:00:00:00.000'),
                                               ('2016:063:16:00:00.000', '2016:064:12:00:00.000'),
                                               ('2016:257:10:00:00.000', '2016:258:12:00:00.000')],
                                 'aach1t':    [('2016:064:01:02:00.000', '2016:064:01:06:00.000')],
                                 'aach2t':    [('2016:064:01:02:00.000', '2016:064:01:06:00.000')],
                                 'oobthr38':  [('2017:074:02:15:00.000', '2017:074:02:22:00.000')],
                                 'oobthr39':  [('2017:074:02:15:00.000', '2017:074:02:22:00.000')],
                                 'oobthr02':  [('2017:312:16:10:00.000', '2017:312:16:12:00.000')],
                                 'oobthr03':  [('2017:312:16:10:00.000', '2017:312:16:12:00.000')],
                                 'oobthr04':  [('2017:312:16:10:00.000', '2017:312:16:12:00.000')],
                                 'oobthr05':  [('2017:312:16:10:00.000', '2017:312:16:12:00.000')],
                                 'oobthr06':  [('2017:312:16:10:00.000', '2017:312:16:12:00.000')],
                                 'oobthr07':  [('2017:312:16:10:00.000', '2017:312:16:12:00.000')],
                                 '4oavobat':  [('2018:285:12:35:00.000', '2018:285:13:15:00.000')],
                                 'ohrmgrd1':  [('2018:217:07:21:00.000', '2018:217:07:23:00.000')],
                                 'oobagrd1': [('2021:059:00:00:00.000', '2021:060:00:00:00.000')],
                                 'oobagrd1': [('2021:067:00:00:00.000', '2021:068:00:00:00.000')],
                                 'oobagrd1': [('2021:296:00:00:00.000', '2021:297:00:00:00.000')],
                                 'oobagrd1': [('2021:360:00:00:00.000', '2021:361:00:00:00.000')],
                                 'oobagrd2': [('2021:059:00:00:00.000', '2021:060:00:00:00.000')],
                                 'oobagrd2': [('2021:067:00:00:00.000', '2021:068:00:00:00.000')],
                                 'oobagrd2': [('2021:296:00:00:00.000', '2021:297:00:00:00.000')],
                                 'oobagrd2': [('2021:360:00:00:00.000', '2021:361:00:00:00.000')],
                                 'oobthr50':  [('2022:069:05:10:00.000', '2022:069:05:15:00.000')],
                                 'oobthr51':  [('2022:069:05:10:00.000', '2022:069:05:15:00.000')],
                                 'tssmax':    [('2022:069:05:10:00.000', '2022:069:05:15:00.000')],
                                 'tssmin':    [('2022:069:05:10:00.000', '2022:069:05:15:00.000')],
                                 'telss_ave': [('2022:069:05:10:00.000', '2022:069:05:15:00.000')],
                                 'ohrthr07':  [('2021:192:09:25:00.000', '2021:192:09:35:00.000')],
                                 'ohrthr08':  [('2021:192:09:25:00.000', '2021:192:09:35:00.000')],
                                 'ee_radial': [('2021:192:09:25:00.000', '2021:192:09:35:00.000')],
                                 'harg':      [('2021:192:09:25:00.000', '2021:192:09:35:00.000')],
                                 'ee_therm':  [('2021:192:09:25:00.000', '2021:192:09:35:00.000')],
                                 '4rt568t':   [('2016:066:18:40:00.000', '2016:066:19:00:00.000')]
                                }




    if ('tel' in setname) or ('oob' in msid) or ('telhs' in msid) or ('ohr' in msid[:3]) \
        or (msid.upper() in healthcheck_msids) or ('4ohtrz' in msid) \
        or ((len(msid) == 3) and (msid[0] == 'p')):
        intervals = [('1999:204:00:00:00.000', '1999:204:12:44:10.000'),
                     ('1999:221:17:20:28.039', '1999:221:17:20:30.039'),
                     ('1999:222:05:59:14.016', '1999:222:05:59:20.016'),
                     ('1999:229:20:16:00.000', '1999:229:20:19:00.000'),
                     ('1999:230:23:53:00.000', '1999:231:01:15:00.000'),
                     ('1999:231:01:07:08.416', '1999:231:01:07:14.416'),
                     ('1999:231:01:08:50.816', '1999:231:01:08:52.816'),
                     ('1999:257:05:35:44.000', '1999:257:07:25:38.000'),
                     ('1999:270:02:00:23.616', '1999:270:02:00:25.616'),
                     ('1999:269:20:27:00.000', '1999:270:07:46:00.000'),
                     ('1999:296:13:58:32.052', '1999:296:13:59:37.052'),
                     ('2000:049:01:12:48.000', '2000:049:02:37:00.000'),
                     ('2000:085:00:34:00.000', '2000:085:00:35:25.000'),
                     ('2004:039:19:18:09.000', '2004:039:19:19:16.000'),
                     ('2008:225:09:25:41.000', '2008:225:09:26:46.000'),
                     ('2008:292:21:22:25.000', '2008:292:21:23:50.000'),
                     ('2011:007:21:25:21.000', '2011:007:21:26:28.000'),
                     ('2011:190:19:22:28.000', '2011:190:20:57:04.000'),
                     ('2012:151:10:30:11.000', '2012:151:12:43:00.000'),
                     ('2014:362:00:05:24.000', '2014:362:00:06:31.000'),
                     ('1999:204:12:43:34.439', '1999:204:12:44:09.239'),
                     ('1999:221:17:15:32.938', '1999:221:17:20:30.138'),
                     ('1999:222:05:54:19.340', '1999:222:05:59:16.540'),
                     ('1999:229:20:17:49.016', '1999:229:20:18:23.816'),
                     ('1999:230:23:53:40.222', '1999:231:01:07:16.551'),
                     ('1999:231:01:07:47.351', '1999:231:01:08:54.951'),
                     ('1999:270:07:34:06.178', '1999:270:07:46:42.578'),
                     ('1999:345:03:21:48.986', '1999:345:05:11:28.411'),
                     ('2000:049:01:07:22.948', '2000:049:02:36:14.381'),
                     ('2008:225:09:26:28.761', '2008:225:09:27:03.561'),
                     ('2010:150:03:36:05.794', '2010:150:03:36:40.594'),
                     ('2011:190:19:21:51.327', '2011:190:20:56:39.330'),
                     ('2012:151:10:29:08.152', '2012:151:12:42:17.472'),
                     ('2015:264:20:37:13.354', '2015:264:22:18:18.799'),
                     ('2016:063:17:11:00.000', '2016:063:17:16:00.000'),
                     ('2016:064:00:40:00.000', '2016:064:04:00:00.000'),
                     ('2016:065:18:29:00.000', '2016:066:19:25:00.000'),
                     ('2018:283:13:54:44.000', '2018:283:14:00:00.000'),
                     ('2018:285:12:35:00.000', '2018:285:13:15:00.000'),
                     ('2020:145:13:00:00.000', '2020:145:15:00:00.000'),
                     ('2020:146:10:00:00.000', '2020:146:21:20:00.000'),
                     ('2021:192:09:25:00.000', '2021:192:09:35:00.000'),
                     ('2022:294:16:25:00.000', '2022:294:16:35:00.000'),
                     ('2022:295:16:45:00.000', '2022:295:17:50:00.000')]
    elif 'eps' in setname:

        intervals = [('2000:034:00:00:00', '2000:035:00:00:00'),
                     ('2000:048:00:00:00', '2000:050:00:00:00'),
                     ('2000:084:00:00:00', '2000:086:00:00:00'),
                     ('2000:106:00:00:00', '2000:107:00:00:00'),
                     ('2009:253:00:00:00', '2009:254:00:00:00'),
                     ('2011:187:00:00:00', '2011:192:00:00:00'),
                     ('2011:299:00:00:00', '2011:300:00:00:00'),
                     ('2012:151:00:00:00', '2012:152:00:00:00'),
                     ('2012:156:00:00:00', '2012:157:00:00:00'),
                     ('2015:244:00:00:00', '2015:245:00:00:00'),
                     ('2015:264:00:00:00', '2015:266:00:00:00'),
                     ('2015:294:00:00:00', '2015:295:00:00:00'),
                     ('2016:063:17:11:00', '2016:063:17:16:00'),
                     ('2018:283:13:54:44.000', '2018:283:14:00:00.000'),
                     ('2022:294:16:25:00.000', '2022:294:16:35:00.000'),
                     ('2022:295:16:45:00.000', '2022:295:17:50:00.000')]

    elif 'isim' in setname:

        intervals = [('1999:257:05:30:00', '1999:257:07:30:00'),
                     ('1999:259:18:16:00', '1999:259:18:17:15'),
                     ('1999:345:03:15:00', '1999:345:05:15:00'),
                     ('2000:049:01:40:00', '2000:049:02:35:00'),
                     ('2012:151:00:00:00', '2012:152:00:00:00'),
                     ('2016:063:17:11:00', '2016:063:17:16:00'),
                     ('2018:283:13:54:44.000', '2018:283:14:00:00.000'),
                     ('2022:294:16:25:00.000', '2022:294:16:35:00.000'),
                     ('2022:295:16:45:00.000', '2022:295:17:50:00.000')]

    elif 'prop' in setname:

        intervals = [('1999:345:03:15:00', '1999:345:05:15:00'),
                     ('2016:063:17:11:00', '2016:063:17:16:00'),
                     ('2000:049:01:12:48.000', '2000:049:02:37:00.000'),
                     ('2018:283:13:54:44.000', '2018:283:14:00:00.000'),
                     ('2022:294:16:25:00.000', '2022:294:16:35:00.000'),
                     ('2022:295:16:45:00.000', '2022:295:17:50:00.000')]

    elif 'pline' in setname:

        intervals = [('1999:345:03:15:00', '1999:345:05:15:00'),
                     ('2016:063:17:11:00', '2016:063:17:16:00'),
                     ('2000:049:01:12:48.000', '2000:049:02:37:00.000'),
                     ('2018:283:13:54:44.000', '2018:283:14:00:00.000'),
                     ('2000:302:05:00:00.000', '2000:302:07:00:00.000'),
                     ('2000:228:05:00:00.000', '2000:228:07:30:00.000'),
                     ('2000:216:23:30:00.000', '2000:217:03:30:00.000'),
                     ('2000:010:19:30:00.000', '2000:010:20:30:00.000'),
                     ('2022:294:16:25:00.000', '2022:294:16:35:00.000'),
                     ('2022:295:16:45:00.000', '2022:295:17:50:00.000')]                     

    else:
        intervals = [('1999:204:00:00:00.000', '1999:204:12:44:10.000'),
                     ('1999:259:17:56:00.000', '1999:259:18:14:10.000'),
                     # ('1999:345:03:20:00.000', '2000:049:05:15:00.000'),
                     ('2000:049:01:37:00.000', '2000:049:02:35:00.000'),
                     ('2011:187:12:25:00.000', '2011:187:12:35:00.000'),
                     ('2012:151:10:31:00.000', '2012:151:10:35:00.000'),
                     ('2016:063:17:11:00.000', '2016:063:17:16:00.000'),
                     ('2018:283:13:54:00.000', '2018:283:14:00:00.000'),
                     ('2022:294:16:25:00.000', '2022:294:16:35:00.000'),
                     ('2022:295:16:45:00.000', '2022:295:17:50:00.000')]

    if msid in list(individual_msid_intervals.keys()):
        badintervals = individual_msid_intervals[msid] # list of tuples
        for badinterval in badintervals: # tuples
            intervals.append(badinterval)

    if msid.upper() in dwell_mode_msids:
        badintervals = get_dwell_mode_intervals()
        for badinterval in badintervals: # tuples
            intervals.append(badinterval)

    intervals.sort()
    
    return intervals


def filter_intervals(times, intervals, stat, extra_pad=0):
    keep = np.array([True] * len(times))

    if 'none' in stat:
        for interval in intervals:
            ind1 = times < (CxoTime(interval[0]).secs - extra_pad)
            ind2 = times > (CxoTime(interval[1]).secs + extra_pad)
            ind = ind1 | ind2 
            keep = keep & ind
    elif '5min' in stat:
        for interval in intervals:
            ind1 = times < (CxoTime(interval[0]).secs - 180 - extra_pad)
            ind2 = times > (CxoTime(interval[1]).secs + 180 + extra_pad)
            ind = ind1 | ind2 
            keep = keep & ind
    elif 'daily' in stat:
        for interval in intervals:
            # ind1 = times < (CxoTime('{}:00:00:00.000'.format(interval[0][:8])).secs - 1 - extra_pad)
            # ind2 = times > (CxoTime('{}:00:00:00.000'.format(interval[1][:8])).secs + 24*3600 + extra_pad)
            ind1 = times < (CxoTime(CxoTime(interval[0]).date[:8]).secs - 1 - extra_pad)
            ind2 = times > (CxoTime(interval[1]).secs + 24*3600 + 1 + extra_pad)

            ind = ind1 | ind2 
            keep = keep & ind
    return keep


def get_keep_ind(times, setname, msid, stat, extra_pad=0):
    
    setname = str(setname).lower()
    stat = str(stat).lower()

    intervals = get_intervals(msid, setname)
        
    keep = filter_intervals(times, intervals, stat, extra_pad)
        
    return keep



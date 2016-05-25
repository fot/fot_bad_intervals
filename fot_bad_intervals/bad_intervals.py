
import numpy as np
from Chandra.Time import DateTime

healthcheck_msids = ['HRMA_AVE','4OAVHRMT','HRMHCHK','OBA_AVE','4OAVOBAT','TSSMAX','TSSMIN',
                     'TABMAX','TABMIN','THSMAX','THSMIN','OHRMGRD3','OHRMGRD6']


def get_intervals(msid, setname=''):
    setname = setname.lower()

    dwell_mode_times = [('2015:244:13:45:00.000', '2015:244:14:45:00.000'),
                        ('2015:294:16:28:00.000', '2015:294:16:36:00.000')]

    individual_msid_intervals = {'3fapsat': [('2011:190:19:43:58.729', '2011:190:19:44:58.729'),
                                             ('2012:151:10:42:03.272', '2012:151:10:43:03.272'),
                                             ('2015:264:20:48:00.000', '2015:264:20:51:00.000')],
                                 'ohrthr15': [('2004:187:15:38:05.785', '2004:187:15:39:05.785')],
                                 'ohrthr21': [('2011:187:12:28:11.387', '2011:187:12:29:11.387')],
                                 'ohrthr22': [('2011:187:12:28:11.387', '2011:187:12:29:11.387')],
                                 'pline10t': [('2003:127:03:50:14.592', '2003:127:03:51:14.592')],
                                 'pline16t': [('2000:318:13:02:33.845', '2000:318:13:03:33.845')],
                                 'tasprwc':  [('2002:296:01:42:08.282', '2002:296:01:43:08.282')],
                                 'tcylaft2': [('2001:005:08:25:32.419', '2001:005:08:26:32.419')],
                                 'tcyz_rw6': [('2000:302:18:58:37.793', '2000:302:18:59:37.793')],
                                 'tep_bpan': [('2003:008:20:09:32.562', '2003:008:20:10:32.562')],
                                 'tfspcmp':  [('2011:299:04:57:52.020', '2011:299:04:58:52.020')],
                                 'tmysada':  [('2011:190:19:47:48.329', '2011:190:19:48:48.329')],
                                 'tmzp_cnt': [('2000:299:16:20:20.183', '2000:299:16:21:20.183')],
                                 'trspotep': [('2003:127:03:50:14.592', '2003:127:03:51:14.592')],
                                 'tsctsf2':  [('2000:055:15:23:18.602', '2000:055:15:24:18.602')],
                                 'cxpnait':  [('2011:299:04:57:00.000', '2011:299:05:00:00.000')],
                                 'toxtsupn': dwell_mode_times,
                                 '4prt5bt': dwell_mode_times,
                                 '4rt585t': dwell_mode_times,
                                 'airu1bt': dwell_mode_times,
                                 'cusoaovn': dwell_mode_times,
                                 'plaed4et': dwell_mode_times,
                                 'pr1tv01t': dwell_mode_times,
                                 'tcylfmzm': dwell_mode_times}


    if ('tel' in setname) or ('oob' in msid) or ('telhs' in msid) or ('ohr' in msid[:3]) \
        or (msid.upper() in healthcheck_msids):
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
                     ('2016:065:18:29:00.000', '2016:066:19:25:00.000')]
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
                     ('2016:063:17:11:00', '2016:063:17:16:00')]

    elif 'isim' in setname:

        intervals = [('1999:257:05:30:00', '1999:257:07:30:00'),
                     ('1999:259:18:16:00', '1999:259:18:17:15'),
                     ('1999:345:03:15:00', '1999:345:05:15:00'),
                     ('2000:049:01:40:00', '2000:049:02:35:00'),
                     ('2016:063:17:11:00', '2016:063:17:16:00')]

    elif 'prop' in setname:

        intervals = [('1999:345:03:15:00', '1999:345:05:15:00'),
                     ('2016:063:17:11:00', '2016:063:17:16:00')]

    else:
        intervals = [('1999:204:00:00:00.000', '1999:204:12:44:10.000'),
                     ('1999:259:17:56:00.000', '1999:259:18:14:10.000'),
                     # ('1999:345:03:20:00.000', '2000:049:05:15:00.000'),
                     ('2000:049:01:37:00.000', '2000:049:02:35:00.000'),
                     ('2011:187:12:25:00.000', '2011:187:12:35:00.000'),
                     ('2012:151:10:31:00.000', '2012:151:10:35:00.000'),
                     ('2016:063:17:11:00', '2016:063:17:16:00')]

    if msid in individual_msid_intervals.keys():
        badintervals = individual_msid_intervals[msid] # list of tuples
        for badinterval in badintervals: # tuples
            intervals.append(badinterval)

    intervals.sort()
    
    return intervals


def get_keep_ind(times, setname, msid, stat):
    
    setname = unicode(setname).lower()
    stat = unicode(stat).lower()

    intervals = get_intervals(msid, setname)
        
    keep = np.array([True] * len(times))
    
    if 'none' in stat:
        for interval in intervals:
            ind1 = times < (DateTime(interval[0]).secs)
            ind2 = times > (DateTime(interval[1]).secs)
            ind = ind1 | ind2 
            keep = keep & ind
    elif '5min' in stat:
        for interval in intervals:
            ind1 = times < (DateTime(interval[0]).secs - 180)
            ind2 = times > (DateTime(interval[1]).secs + 180)
            ind = ind1 | ind2 
            keep = keep & ind
    elif 'daily' in stat:
        for interval in intervals:
            ind1 = times < DateTime('{}:00:00:00.000'.format(interval[0][:8])).secs - 1
            ind2 = times > DateTime('{}:00:00:00.000'.format(interval[1][:8])).secs + 24*3600
            ind = ind1 | ind2 
            keep = keep & ind
        
    return keep

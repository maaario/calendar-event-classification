from datetime import datetime, timedelta, date
import re
import unidecode
import numpy as np
import matplotlib.pyplot as plt

from icalendar.cal import Calendar
from icalendar.prop import vDatetime

#######################
# READ CALENDAR FILES #
#######################

def clean_summary(summary):
    cs = unidecode.unidecode(summary)
    cs = re.sub(r'[^a-z0-9 ]', ' ', cs.strip().lower())
    cs = ' '.join([word.strip() for word in cs.split()])
    return cs

def calendar_to_events(cal):
    events = []
    for component in cal.walk("VEVENT"):
        summary = component.get('summary')
        summary = clean_summary(summary)

        start = component.get('dtstart').dt
        end = component.get('dtend').dt
        duration = end - start

        repeated = component.get('rrule') is not None

        # Don't include whole day events and repeated evenrs for now
        if not repeated and duration < timedelta(days=1):
            event = {'summary': summary, 'start': start, 'duration': duration, 'label': ''}
            events.append(event)
    return events

def global_label(events, label):
    for event in events:
        event['label'] = label

events = []
with open('calendar.ics', 'r') as cal_file:
    cal = Calendar.from_ical(cal_file.read())
    new_events = calendar_to_events(cal)
    global_label(new_events, 'personal')
    events.extend(new_events)

with open('school.ics', 'r') as cal_file:
    cal = Calendar.from_ical(cal_file.read())
    new_events = calendar_to_events(cal)
    global_label(new_events, 'school')
    events.extend(new_events)

with open('sleep.ics', 'r') as cal_file:
    cal = Calendar.from_ical(cal_file.read())
    new_events = calendar_to_events(cal)
    global_label(new_events, 'sleep')
    events.extend(new_events)

print(len(events), 'events loaded.')

##################################
# CREATE FEATURES FOR EACH EVENT #
##################################

def datetime_to_features(dt):
    # https://datascience.stackexchange.com/questions/2368/machine-learning-features-engineering-from-date-time-data
    # TODO: look into data to find best attributes

    day_number = (dt.date() - date(2014, 1, 1)).days
    weekday = dt.weekday()
    time_of_day_minutes = dt.hour * 60 + dt.minute

    return [day_number, weekday, time_of_day_minutes]

def event_to_features(event):
    features = []
    features.append(event['summary'])
    features.extend(datetime_to_features(event['start']))
    features.append(event['duration'].total_seconds() / 60)
    features.append(event['label'])

    return features

def print_events(events):
    columns = ['summary', 'day_number', 'weekday', 'minutes_of_day', 'duration_minutes', 'label']
    lines = [' '.join(columns)]

    for event in events:
        features = event_to_features(event)
        features[0] = '"{}"'.format(features[0])
        features[-1] = '"{}"'.format(features[-1])
        
        lines.append(' '.join([str(x) for x in features]))

    return '\n'.join(lines) 

def events_to_matrix(events):
    label_to_int = {
        'personal': 0,
        'school': 1,
        'sleep': 2, 
    }

    mat = np.zeros([len(events), 5])
    for i, event in enumerate(events):
        features = event_to_features(event)
        features[-1] = label_to_int[features[-1]]
        mat[i, :] = np.array(features[1:])
    return mat

data = events_to_matrix(events)
print(data)

with open('processed_events.txt', 'w') as f:
    f.write(print_events(events))

#############
# VISUALIZE #
#############

hours_of_day_covered = dict()
for event in data:
    day = event[0]
    end = event[2] + event[3]
    duration = (event[3] - (end - (24 * 60) if end > (24 * 60) else 0)) / 60

    covered = hours_of_day_covered.get(day, 0)
    hours_of_day_covered[day] = covered + duration

    if end > 24 * 60:
        duration = (covered + end - 24 * 60) / 60
        covered = hours_of_day_covered.get(day + 1, 0)
        hours_of_day_covered[day + 1] = covered + duration

hours_of_day_covered = np.array(list(hours_of_day_covered.items()))

# plt.figure(1)
# plt.scatter(hours_of_day_covered[:, 0], hours_of_day_covered[:, 1])
# plt.title('Covered time vs. day number')

# plt.figure(2)
# plt.hist(hours_of_day_covered[:, 1])
# plt.title('Hours covered per day')
# plt.show()

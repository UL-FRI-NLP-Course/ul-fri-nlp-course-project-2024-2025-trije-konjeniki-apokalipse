from . import cache

import pandas as pd
from striprtf.striprtf import rtf_to_text

import glob
import os
import re
from zoneinfo import ZoneInfo


_file_dir = os.path.dirname(__file__)
_parsed_reports_path = os.path.join(_file_dir, '../data/parsed/traffic_reports.csv')



def _read_one_traffic_report(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        rtf_content = f.read()
        text = rtf_to_text(rtf_content)
    return text



def _read_traffic_reports(paths: list[str]):
    reports = []
    for i, path in enumerate(paths):
        print(f'Reading traffic report {i+1} of {len(paths)}', end='\r')
        report_cur = _read_one_traffic_report(path)
        reports.append(report_cur)
    print(f'Finished reading {len(reports)} traffic news reports.')
    return reports



def _parse_traffic_report(report: str):
    lines = [line.strip() for line in report.split('\n') if line.strip()]
    header = lines[0].replace('\t', '    ').replace('-', ' ').replace('..', '.')

    # Parse date, hardcoded fixes for some dataset inconsitencies -_-
    header = header.replace('10. 10. 22', '10. 10. 2022')
    date_pattern = re.compile(r'\d{1,2}[\. ] ?\d{1,2}[\. ] ?\d{4}')
    date_match = date_pattern.search(header)
    date_str = date_match.group() if date_match else None

    # Parse report type and main content
    report_type = header[:date_match.span()[0]].strip() if date_match else None
    content = '\n'.join(lines[1:])

    # Parse time
    time_pattern = re.compile(r'\b\d{1,2}[.:]\d{2}\b')
    time_pos = date_match.span()[1] if date_match else 0
    time_match = time_pattern.search(header, pos=time_pos)
    time_str = time_match.group() if time_match else None

    # Parse programme
    i_time_end = time_match.span()[1] if time_match else None
    programme = header[i_time_end:].strip() if i_time_end else None

    return {
        'type': report_type,
        'date': date_str,
        'time': time_str,
        'programme': programme,
        'content': content
    }



@cache.csv(_parsed_reports_path)
def _load_traffic_reports(paths: list[str]):
    reports = _read_traffic_reports(paths)
    parsed_reports = [_parse_traffic_report(report) for report in reports]
    df_reports = pd.DataFrame(parsed_reports)
    df_reports['text'] = reports
    df_reports['path'] = paths

    # Remove empty documents
    mask_empty = (df_reports.text.str.strip().str.len() < 10)
    df_reports = df_reports[~mask_empty]
    print(f'Found and removed {mask_empty.sum()} ({mask_empty.mean()*100:.2f}%) empty documents.')

    # Store date and time as datetime objects
    df_reports['time'] = df_reports['time'].str.replace('.', ':')
    df_reports['date'] = df_reports['date'].str.replace('.', ' ').replace(r'\s+', '. ', regex=True)
    df_reports['datetime'] = pd.to_datetime(df_reports['date'] + ' ' + df_reports['time'], dayfirst=True, errors='coerce')
    print(f'Parsed datetime successfully for {df_reports.datetime.notna().sum()} ({df_reports.datetime.notna().mean()*100:.2f}%) documents. Removing the rest.')
    df_reports = df_reports[df_reports.datetime.notna()]
    df_reports = df_reports.sort_values('datetime')

    # Make types of reports consistent
    types_map = {
        'Prometne informacije': ['Prometne informacije', 'Prometna informacija', 'Prometne informacije.', 'Prometna informacija.', 'Prometne informacije:', 'Prometne informacije      11.', 'Prometne informacije        10.', 'Podatki o prometu.', 'Prometne informacijje', 'Podatki o prometu', 'Prometne informacije 1. program', 'rometne informacije', 'Pometne informacije', 'DIO ZAČETEK Prometna informacija', 'lPrometne informacije', 'informacije', '.Prometna informacija', 'metne informacije', 'Podatki o prometu:', 'PRAVA Prometne informacije'],
        'Nove prometne informacije': ['NOVE Prometne informacije', 'NOVA Prometne informacije', 'NOVE prometne informacije', 'NOVEPrometne informacije', 'NOVA Prometna informacija', 'NOVE Prometna informacija', 'NOVAPrometne informacije', 'NOVE NOVE Prometne informacije', 'NOVO Prometne informacije', 'Nove Prometne informacije', 'NOVE Prometne informacije:', 'NOVA  Prometne informacije', 'Nove prometne informacije.', 'NAJNOVEJŠE prometne informacije', 'NOVA  do   Prometne informacije', 'NOVE  informacije', 'NOVA +Prometne informacije', 'NOVE2 Prometne informacije', 'NOva Prometne informacije', 'ometne informacije', 'NOVAPrometna informacija', 'NOVE Prometne informacije.', 'NOVI Podatki o prometu', 'NOVA      Prometne informacije', 'Nova Prometne informacije'],
        'Nujne prometne informacije': ['Nujna prometna informacija', 'Nujna prometna informacija.', 'NUJNA Prometna informacija', 'NUJNO! Prometne informacije', 'Nujne prometne informacije', 'ujna prometna informacija.', 'NOVA Nujna prometna informacija'],
        None: ['PROMET RD,', '9:15', '7.30', '8.15']
    }

    for new_type, old_types in types_map.items():
        mask = df_reports['type'].isin(old_types)
        df_reports.loc[mask, 'type'] = new_type

    return df_reports



def load_traffic_reports(path: str):
    report_paths = glob.glob(path, recursive=True)
    df_reports = _load_traffic_reports(report_paths)
    df_reports['datetime'] = pd.to_datetime(df_reports['datetime'])
    df_reports['date'] = df_reports['datetime'].dt.date
    df_reports['time'] = df_reports['datetime'].dt.time
    return df_reports



def load_promet_si(path: str):
    # In the excel file, data is split by years into sheets, to read them all, use `sheet_name=None`
    sheet_to_df = pd.read_excel(path, sheet_name=None)
    df = pd.concat(sheet_to_df.values(), ignore_index=True)

    # Convert timestamp to local timezone (as used in the traffic reports). This will also handle daylight saving time transformations.
    zone = ZoneInfo('Europe/Ljubljana')
    df['Datum'] = df['Datum'].dt.tz_localize('UTC').dt.tz_convert(zone).dt.tz_localize(None)
    return df

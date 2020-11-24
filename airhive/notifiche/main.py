#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
from datetime import datetime
import logging
import json
import requests
import warnings

import numpy as np
import pandas as pd
import pymysql
import urllib.request

from pyfcm import FCMNotification
from scipy.spatial import cKDTree
from sqlalchemy import create_engine

import tools

warnings.simplefilter(action='ignore', category=FutureWarning)
push_service = FCMNotification(api_key="AAAAVIVJ3xo:APA91bFvnWh-uDWoRPb8PUDITkVvNQUHxTR8T0z964R-3GdIZJa0BwoEyEt3Xv2RNPXydBobC5PwJRT8blozBhUb7vKzbvmfMxrCRdR2S6X48QizXetG-xV4JRQ1xLOlpAuUyrvTVwim")
ora_adesso = datetime.now().hour

controlla_anomalie = tools.controlla_anomalie

def get_data():
    """Legge il database per prendere i dati dei sensori"""
    #TODO quando collegato da aggiustare coi dati veri, non gli stessi ripetuti
    db_connection_str = 'mysql+pymysql://sito:FucoFico1998@house.zan-tech.com:1433/hive_db'
    db_connection = create_engine(db_connection_str)
    return pd.read_sql('SELECT pm10, temp, umi, prec, vento, no2, o3, tempo FROM sensori ORDER BY tempo DESC, id_sensore ASC LIMIT 1000', con=db_connection)

def get_posizione_sensori(location_sensori):
    """Legge il database per prendere le coodinate dei sensori della città scelta"""
    db_connection_str = 'mysql+pymysql://sito:FucoFico1998@house.zan-tech.com:1433/hive_db'
    db_connection = create_engine(db_connection_str)
    return pd.read_sql('SELECT lat, lng FROM id_sensori{}'.format(location_sensori), con=db_connection)

def get_users():
    """Prende i dati degli utenti dal server"""
    with urllib.request.urlopen("https://www.airhive.it/php/getDevices.php?withPos=true&key=kbdq2308o24jiGKlmkjguydrbNHNUIYGjh378jhwqgxmiljhqdknbx972khdnibg3DJNDd1n8ygqf") as url:
        df = pd.DataFrame(json.loads(url.read().decode())["data"])
    df.columns = ["tkn", "lat", "lng", "hl"]
    df.lat = pd.to_numeric(df.lat)
    df = df[df.lat != 0].reset_index(drop=True)
    df.lng = pd.to_numeric(df.lng)
    return df

def lingue(choosen_one_s, livello_pericolo):
    """Seleziona le lingue, se non ci sono è inglese."""
    lingue = ["IT", "DE", "FR", "ES"]
    utenti_lingua = [choosen_one_s.hl == lingua for lingua in lingue]
    # Quelli che non sono nelle lingue sopra
    utenti_lingua.append(pd.concat(utenti_lingua, axis=1).sum(axis=1) == 0)
    Traduzioni = namedtuple('Traduzioni', 'message_title message_body')
    if livello_pericolo == "df_2":
        italiano = Traduzioni("Inquinamento insolito", "Rilevati livelli insolitamente alti di inquinamento nell'area, prestare particolare attenzione.")
        tedesco = Traduzioni("Ungewöhnlicher Verschmutzungsgrad", "Beachten Sie bitte, dass ungewöhnlich viele Schadstoffe festgestellt werden.")
        francese = Traduzioni("Niveau de pollution inhabituel", "Niveau inhabituellement élevé de polluants détectés, veuillez faire attention.")
        spagnolo = Traduzioni("Nivel de contaminación inusual", "Nivel inusualmente alto de contaminantes detectados, tenga cuidado.")
        altre_lingue = Traduzioni("Unusual pollution level", "Unusually high level of pollutants detected, please be careful.")
    else:
        italiano = Traduzioni("Inquinamento pericoloso", "Rilevati livelli estremamenti alti di inquinamento nell'area, prestare particolare attenzione.")
        tedesco = Traduzioni("Gefährliche Verschmutzung", "Sehr hohe Schadstoffkonzentration, bitte besondere Aufmerksamkeit zu widmen.")
        francese = Traduzioni("Niveau de pollution dangereux", "Très haute concentration de polluants détectés, soyez particulièrement prudent.")
        spagnolo = Traduzioni("Nivel de contaminación peligrosa", "Nivel extremadamente alto de contaminantes detectados, tenga especial cuidado.")
        altre_lingue = Traduzioni("Dangerous pollution level", "Extremely high level of pollutants detected, please be especially careful.")
    return utenti_lingua, [italiano, tedesco, francese, spagnolo, altre_lingue], lingue.append("EN")

def prep_log():
    """ Come visto in https://docs.python.org/3/howto/logging-cookbook.html """
    # create logger with 'errori_notifiche'
    logger = logging.getLogger('errori_notifiche')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('notifiche.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def seleziona_e_invia(store, sensori_selezionati, df_users, nome_database, location_sensori, albero):
    """Se ci sono anomalie fuori dalle 3 std
    Le coordinate dei sensori fuori dalle 3 std:"""
    # Tempo tra le notifiche
    tempo_tra_notifiche = 6
    # Obbligatorio qui per evitare i Nan
    coo_utenti = [(lat,lon) for lat,lon in zip(df_users.lat, df_users.lng)]
    # Migliore con molti dati: coo_sensori = list(sensori_selezionati.itertuples(index=False, name=None))
    coo_sensori = [(lat,lon) for lat,lon in zip(sensori_selezionati.lat, sensori_selezionati.lng)]
    idx, tree = verifica_distanza(loc_users=coo_utenti, points_sensori=coo_sensori, albero=albero)
    # Selezione utenti vicini
    choosen_one_s = df_users.loc[idx]
    # Segno l'ora a cui li ho avvisati come 50, così se aspetto n ore < 56 per avvisare comunque ho {x in 24} - -50 > n
    tempo_sicurezza = -50
    choosen_one_s["ora_avviso"] = tempo_sicurezza
    # Controlla chi ho già avvisato
    df_store = store[nome_database]
    try:
        # Prendo quelli scelti questa volta e ci metto l'ora a cui erano stati scelti l'altra volta
        # Concateno df_store e i prescelti, aggiungo un indice, cerco i duplicati e copio la vecchia ora di avviso nella nuova
        gia_notificati = pd.concat([df_store, choosen_one_s], keys=['s1', 's2']).tkn.duplicated(keep=False)
        choosen_one_s.loc[gia_notificati["s2"], "ora_avviso"] = df_store[gia_notificati["s1"]].ora_avviso
    except AttributeError:
        # Se df_store è vuoto non ha la colonna tkn
        None
    except ValueError:
        # Se df_store è vuoto non posso compararlo
        None
    except KeyError as e:
        # Se l'errore è in s2 choosen_one_s è vuoto e non c'è nessuno da avvisare
        if e.args[0] == "s2":
            # Restituisce tutti quelli avvisati da poco ma non questa volta
            return df_store[(ora_adesso - df_store.ora_avviso) < tempo_tra_notifiche]
        None
    # Seleziono quelli che non avviso da 6 ore
    # 6 le ore che aspetto prima di avvisare di nuovo
    overtime = choosen_one_s[(ora_adesso - choosen_one_s.ora_avviso) > tempo_tra_notifiche]
    if not choosen_one_s.equals(df_store) or overtime.size > 0:
        # Salva gli utenti che sto per avvisare e correggi gli orari
        choosen_one_s.loc[choosen_one_s.ora_avviso == tempo_sicurezza, "ora_avviso"] = ora_adesso
        store.put(nome_database, choosen_one_s)
        # Seleziona i giusti gruppi di utenti, se non c'è la lingua manda in inglese
        utenti, traduzioni, lingue = lingue(choosen_one_s, nome_database)
        map(lambda traduzione, utenti_lingua, lingua: send_mess(
            registration_ids=overtime.loc[utenti_lingua].tkn.values.tolist(), 
            message_title=traduzione.message_title, 
            message_body=traduzione.message_body, 
            location_sensori=location_sensori, 
            lingua=lingua
            ), 
            zip(traduzioni, utenti, lingue)
        )
    # Tutti quelli da non avvisare
    return df_store[(ora_adesso - df_store.ora_avviso) < tempo_tra_notifiche], tree

def send_mess(registration_ids, message_title, message_body, location_sensori, lingua):
    """Invia notifica, vedi https://github.com/olucurious/PyFCM/blob/master/pyfcm/fcm.py"""
    # Per i test return qui
    return
    result = push_service.notify_multiple_devices(
        registration_ids=registration_ids, 
        message_title=message_title, 
        message_body=message_body, 
        time_to_live=3360, 
        color="#ffea00"
    )
    requests.get(
        "https://www.airhive.it/php/fireAlert.php?key=kwbfnjr398y8gqrmo2KBuf84349gaL&t={}&m={}&r={}&hl={}".format(
            message_body, 
            message_title, 
            location_sensori, 
            lingua
        ).replace(" ", "%20")
    )

def verifica_distanza(loc_users, points_sensori, albero):
    """Restituisce gli indici degli utenti con distanza minore di 0.5
    Quali sono gli utenti più vicini a quei sensori?
    n_jobs=-1 usa tutti i processori"""
    # Invertendo e usando sensori al posto di users potrei salvare il tree finchè i sensori non cambiano
    if albero == None:
        tree = cKDTree(loc_users)
    else:
        tree = albero
    idx = tree.query_ball_point(points_sensori, 0.5, n_jobs=-1)
    return np.unique(np.concatenate(idx).ravel()), tree

def main(numero_sensori, location_sensori):
    try:
        # I log
        logger = prep_log()
        df_users = get_users()
        df_data = get_data()
        pos_sensori = get_posizione_sensori(location_sensori=location_sensori)
        # Mi servono molti valori per media e std ma devo controllare solo l'ultima rilevazione
        ultime_rilevazioni = df_data[-numero_sensori:]
        media = df_data.pm10.mean()
        dev_std = df_data.pm10.std()
        # Cerca anomalie fuori da 2 std
        selezione = pd.Series(
            controlla_anomalie(
                df=ultime_rilevazioni.pm10.values, 
                media=media, 
                dev_std=dev_std, 
                numero_deviazioni=1.96
            )
        )
        if not selezione.any():
            # Se non ci sono anomalie ho finito
            return
        # Controlla tra le anomalie std2 se ci sono anomalie std3
        selezione_devstd3 = pd.Series(
            controlla_anomalie(
                df=ultime_rilevazioni.reset_index(drop=True).pm10[selezione].values, 
                media=media, 
                dev_std=dev_std, 
                numero_deviazioni=3
            )
        )
        #Il risultato ha indici a zero, reindicizza
        pos_sensori_selezione = pos_sensori[selezione]
        selezione_devstd3.index = pos_sensori_selezione.index
        # Seleziona solo la prima volta i sensori fuori da std3 tra gli std2
        sensori_fuori_std3 = pos_sensori_selezione[selezione_devstd3]
        # Selezione degli elementi fuori da 2 std ma non fuori da 3 std
        fuori_2_dentro_3 = (selezione & ~selezione_devstd3)
        # Apre solo una volta questo database
        store = pd.HDFStore('store_users_{}.h5'.format(location_sensori))
        # Un df vuoro per attivare sotto se non parte il primo if
        avvisati = pd.DataFrame([])
        # Variabile per non ricalcolare l'albero
        albero = None
        if selezione_devstd3.any():
            """Se ci sono anomalie fuori dalle 3 std
            Le coordinate dei sensori fuori dalle 3 std:"""
            avvisati, tree = seleziona_e_invia(
                store=store, 
                sensori_selezionati=sensori_fuori_std3, 
                df_users=df_users, 
                nome_database="df_3", 
                location_sensori=location_sensori, 
                albero=albero
            )
        if fuori_2_dentro_3.any():
            """Se ci sono anomalie fuori dalle 2 std che non sono
            anche fuori dalle 3 std"""
            # Prima la condizione più plausibile per aumentare la velocità
            if avvisati.empty:
                seleziona_e_invia(
                    store=store, 
                    sensori_selezionati=pos_sensori[fuori_2_dentro_3], 
                    df_users=df_users, 
                    nome_database="df_2", 
                    location_sensori=location_sensori, 
                    albero=tree
                )
            else:
                # Prendo gli indici di quelli che non ho già avvisato nel precedente overtime
                #TODO probabilmente c'è un modo più veloce usando il db in hd5
                idx = pd.concat(
                        [avvisati.tkn, df_users.tkn], 
                        keys=['over', 'users']
                    ).drop_duplicates(keep=False)["users"].index
                seleziona_e_invia(
                    store=store, 
                    sensori_selezionati=pos_sensori[fuori_2_dentro_3], 
                    df_users=df_users.loc[idx].reset_index(drop=True), 
                    nome_database="df_2", 
                    location_sensori=location_sensori, 
                    albero=tree
                )
        store.close()
    except Exception as e:
        print(e)
        logger.info("Errore: ", e)
        store.close()



if __name__ == "__main__":
    numero_sensori = 100
    posti = ["merano", "milano", "vicenza", "bolzano"]
    map(lambda posto: main(numero_sensori=numero_sensori, location_sensori=posto), posti)

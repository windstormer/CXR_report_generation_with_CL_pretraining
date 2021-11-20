#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 03:19:51 2021

@author: nlp
"""

import paramiko
import os


t = paramiko.Transport(('140.114.93.130', 22))
t.connect(username = 'whshen', password = 'a05040602')
sftp = paramiko.SFTPClient.from_transport(t)

for fname in sorted(os.listdir('./model')):
    print(fname)
    
    localpath='/model/{}/CNN_model_400.pth'.format(fname)
    remotepath='/model/{}/CNN_model_400.pth'.format(fname)
    
    sftp.put(localpath,remotepath)
    
    break
    
t.close()

# username = 'whshen'
# password = 'a05040602'
# hostname = '114.140.93.130'
# port = 22

# try:
#     client = paramiko.SSHClient()
#     client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     client.connect(hostname, port, username, password)
#     t = client.get_transport()
#     sftp=paramiko.SFTPClient.from_transport(t)
#     d = sftp.stat("/Users/allen/Dropbox/python/ssh.txt")
#     print (d)
#     client.exec_command('cd /Users/allen/Dropbox/python')
#     stdin, stdout, stderr = client.exec_command('ls -al')
#     result = stdout.readlines()
#     print (result)
# except Exception:
#     print ('Exception!!')
#     raise

import time
import os
import socket
import struct
import sys
import traceback

def recvfile():
    #获取计算机名称
    hostname = socket.gethostname()
    #获取本机ip
    ip_addr = socket.gethostbyname(hostname)
    # 明确配置变量
    ip_port = 8080
    # print("ip:",ip_addr, ip_port)
    back_log = 5
    # argc = len(sys.argv)
    savedir = "./"
    # assert os.path.exists(savedir), "save dir not exits"
    id_ = 0x1234

    paramdict = {"ip_addr":ip_addr,"ip_port":ip_port,"id":id_}
    for i, (param_k, param_v) in enumerate(paramdict.items()):
        if param_k == "id":
            print(f"{i + 1}: {param_k:8s}({hex(param_v)})")
        else:
            print(f"{i + 1}: {param_k:8s}({param_v})")

    while 1:
        try:
            chs = input("input a choice：")

            chs = int(chs)
            if chs == 1:
                vas = input("input ip_addr: ")
                ip_addr = vas
            elif chs == 2:
                vas = input("input ip_port: ")
                ip_port = int(vas)
            elif chs == 3:
                vas = input("input id_: ")
                id_ = eval(f"0x{vas}")
            else:
                break
        except:
            print("input error")
            continue

    paramdict = {"ip_addr":ip_addr,"ip_port":ip_port,"id":id_}
    for i, (param_k, param_v) in enumerate(paramdict.items()):
        if param_k=="id":
            print(f"{param_k}: {hex(param_v)} ", end='')
        else:
            print(f"{param_k}: {param_v} ", end='')
    print()
    # print(ip_addr, ip_port, id_==0x1234)
    # exit()
    # buffer_size = 1024
    # 创建一个TCP套接字
    ser = socket.socket(socket.AF_INET,socket.SOCK_STREAM)   # 套接字类型AF_INET, socket.SOCK_STREAM   tcp协议，基于流式的协议
    ser.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)  # 对socket的配置重用ip和端口号
    # 绑定端口号
    ser.bind((ip_addr, ip_port))  #  写哪个ip就要运行在哪台机器上
    # 设置半连接池
    ser.listen(back_log)  # 最多可以连接多少个客户端

    while 1:
        print("server start")
        # 阻塞等待，创建连接
        conn,address = ser.accept()  # 在这个位置进行等待，监听端口号
        while 1:
            try:
                # 接受套接字的大小，怎么发就怎么收
                # 接受id，判断是否可以接受数据
                fmt = f">H"
                buffer_size = struct.calcsize(fmt)
                msg = conn.recv(buffer_size)
                # 判断是否数据为空
                if not msg:
                    # 断开连接
                    conn.close()
                    break
                # 解析id
                id = struct.unpack(fmt, msg)[0]
                # 给客户端发信息flag，是否继续发送
                flag = 1 if id == id_ else 0
                flag_msg = struct.pack(">I", flag)
                conn.send(flag_msg)
                if flag == 0:
                    print("id error")
                    break

                # buffer_size = 1024 # 也可以
                # 接收一些基本信息，修改时间，路径长度，文件长度，最大发送长度
                fmt = f">4I"
                buffer_size = struct.calcsize(fmt)
                # print("buffer_size", buffer_size)

                msg = conn.recv(buffer_size)
                # print("msg1", len(msg))

                mtime, relpath_len, buffer_size, RECV_BUF_SIZE = struct.unpack(fmt, msg)
                # print("id:", id, mtime, relpath_len, buffer_size, RECV_BUF_SIZE)

                # 接收路径
                msg = conn.recv(relpath_len)
                rel_path = struct.unpack(f">{relpath_len}s", msg)[0].decode('utf-8')
                # print("msg2", len(msg))
                # print(rel_path)
                savepath = os.path.join(savedir, rel_path)
                # 给客户端发送flag信息， 是否让其发送文件
                flag = 1
                if os.path.exists(savepath):
                    mtime_ = int(os.stat(savepath).st_mtime)
                    if mtime_ >= mtime:
                        print(f"pass file: {os.path.abspath(savepath)}")
                        # print(f"passed file: {savepath:<50s} oldfile_time: {mtime_}, sendfile_time: {mtime}")
                        flag = 0
                flag_msg = struct.pack(">I", flag)
                conn.send(flag_msg)

                # 接收客户端发的文件
                if flag:
                    # print("flag", flag)
                    # msg = con.recv(buffer_size)
                    if buffer_size>RECV_BUF_SIZE:
                        recv_size = 0
                        recv_msg = b''
                        while recv_size < buffer_size:
                            # recv_msg += conn.recv(RECV_BUF_SIZE)
                            recv_msg += conn.recv(min(RECV_BUF_SIZE, buffer_size-recv_size))
                            recv_size = len(recv_msg)
                    else:
                        recv_msg = conn.recv(buffer_size)

                    # print("msg3", len(recv_msg))
                    buffer = struct.unpack(f"{buffer_size}s", recv_msg)[0]
                    # print("flag",flag, len(buffer), buffer_size)
                    # if flag>0:
                        # decode(buffer)
                    filedir = os.path.dirname(savepath)
                    if not os.path.exists(filedir):
                        os.makedirs(filedir)
                    # print(path)
                    with open(savepath, 'wb') as f:
                        f.write(buffer)  # 与readlines配对
                    print(f"recv file: {os.path.abspath(savepath)}")
                    # print(f"recved file: {savepath:<50s} bufsize: {buffer_size:<10d}, recvsize: {len(recv_msg):<10d}")
                continue
            except Exception as e:
                print(e)
                exc = traceback.format_exc()
                print(exc)
                break
    # 关闭服务器
    # ser.close()

def sendfile():
    epath = r"" # 默认路径
    epathlist = []
    inextlist = []
    exextlist = []
    excludedir = []

    argc = len(sys.argv)
    # 获取计算机名称
    hostname = socket.gethostname()
    # 获取本机ip
    ip_addr = socket.gethostbyname(hostname)
    ip_port = 8080
    bufsize = 4096000
    id_ = 0x1234
    start = 2
    if argc > 1:
        ip_addr = sys.argv[1]
    if argc > 2:
        ip_port = int(sys.argv[2])
    if argc > 3:
        id = int(sys.argv[3])
    if argc > 4:
        start = int(sys.argv[4])
    if argc > 5:
        bufsize = int(sys.argv[5])
    if argc > 6:
        epath = sys.argv[6]
        epathlist.append(epath)

    # paramdict = {"ip_addr": ip_addr, "ip_port": ip_port, "id": id_, "start": start, "bufsize": bufsize, "epath": epath}
    listdict = {"epath": epathlist, "inext": inextlist, "exext": exextlist, "exdir": excludedir}

    paramdict = {"ip_addr": ip_addr, "ip_port": ip_port, "id": id_, "start": start, "bufsize": bufsize}

    if len(epath) > 0:
        epathlist.append(epath)
    for i, (param_k, param_v) in enumerate(paramdict.items()):
        if param_k == "id":
            print(f"{i + 1}: {param_k:8s}({hex(param_v)})")
        else:
            print(f"{i + 1}: {param_k:8s}({param_v})")

    print("6: add\n7: rm")

    while 1:
        try:
            chs = input("input a choice：")
            chs = int(chs)
            if chs == 1:
                vas = input("input ip_addr: ")
                ip_addr = vas
            elif chs == 2:
                vas = input("input ip_port: ")
                ip_port = int(vas)
            elif chs == 3:
                vas = input("input id_: ")
                id_ = eval(f"0x{vas}")
            elif chs == 4:
                vas = input("input start: ")
                start = int(vas)
            elif chs == 5:
                vas = input("input bufsize: ")
                bufsize = int(vas)
            elif chs == 6:
                ld_k = list(listdict.keys())
                print(ld_k)
                chs2 = input(f"input listindex to add <{len(listdict)}: ")

                ld_v = listdict[ld_k[int(chs2)]]

                while 1:
                    # print("input a digit to break")
                    vas = input("input value, digit to break: ")
                    if vas.isdigit():
                        break
                    if ld_k[int(chs2)] == "epath":
                        if os.path.exists(vas):
                            epathlist.append(vas)
                            print(f"epath:{os.path.abspath(vas)}")
                            print(f"epathlist: {epathlist}")
                        else:
                            print("path not exist")
                            continue
                    else:
                        ld_v.append(vas)
                print(listdict)

            elif chs == 7:
                ld_k = list(listdict.keys())
                print(ld_k)
                chs2 = input(f"input listindex to rm <{len(listdict)}: ")

                ld_v = listdict[ld_k[int(chs2)]]
                if len(ld_v) == 0:
                    print("nothing to remove")
                    continue

                while 1:
                    print(ld_v)
                    vas = input(f"input rm index(int), -1 to break<{len(ld_v)}: ")
                    vas = int(vas)
                    if vas < 0:
                        break
                    ld_v.pop(vas)

                print(listdict)
            else:
                break
        except:
            print("input error")
            continue

    paramdict = {"ip_addr": ip_addr, "ip_port": ip_port, "id": id_, "start": start, "bufsize": bufsize}

    for i, (param_k, param_v) in enumerate(paramdict.items()):
        if param_k == "id":
            print(f"{param_k}: {hex(param_v)} ", end='')
        else:
            print(f"{param_k}: {param_v} ", end='')
    print()
    if len(epathlist)==0:
        epathlist.append('./')
    print(listdict)

    # assert len(epathlist) > 0


    SEND_BUF_SIZE = bufsize  # if bufsize != None else 40960000
    # SEND_BUF_SIZE = 4096
    p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # print("ip:", ip_addr, ip_port)
    p.connect((ip_addr, ip_port))

    for epath in epathlist:
        if not os.path.exists(epath):
            print(f"path {epath} not exits")
            continue
        epath = os.path.abspath(epath)
        epath = epath.replace('\\', '/')
        epath_len = len(epath.split('/'))
        start = epath_len - 1 if start == None else start
        for roots, dirs, files in os.walk(epath):
            roots = roots.replace('\\', '/')
            rootslist = roots.split('/')
            rootslist_len = len(rootslist)
            rel_len = rootslist_len - epath_len
            # print("rel_len", rel_len)
            dir_1 = rootslist[-1] if rel_len > 0 else None
            dir_2 = rootslist[-2] if rel_len > 1 else None
            dir_3 = rootslist[-3] if rel_len > 2 else None
            # print(dir_1, dir_2, dir_3)
            if dir_1 in excludedir or dir_2 in excludedir or dir_3 in excludedir:
                continue
            for file in files:
                ext = file.split('.')[-1]
                if ext in exextlist:
                    continue
                if ext in inextlist or len(inextlist) == 0:

                    filepath = os.path.join(roots, file).replace("\\", '/')

                    pathlist = filepath.split('/')
                    rel_path = '/'.join(pathlist[start:]).encode('utf-8')
                    relpath_len = len(rel_path)
                    with open(filepath, 'rb') as f:
                        buffer = f.read()
                        buffer_size = len(buffer)
                        mtime = int(os.stat(filepath).st_mtime)
                        # 发送id
                        fmt = f">H"
                        msg = struct.pack(fmt, id_)
                        p.sendall(msg)
                        # 接受服务器反馈
                        flagsize = struct.calcsize(">I")
                        recv_msg = p.recv(flagsize)
                        flag = struct.unpack(">I", recv_msg)[0]

                        if flag == 0:
                            print("id error")
                            exit(-1)
                        # print("flag", flag)

                        # 发送一些基本信息，修改时间，路径长度，文件长度，最大发送长度
                        fmt = f">4I"
                        msg = struct.pack(fmt, mtime, relpath_len, buffer_size, SEND_BUF_SIZE)
                        p.sendall(msg)

                        # 发送路径
                        fmt = f">{relpath_len}s"
                        msg = struct.pack(fmt, rel_path)
                        p.sendall(msg)
                        # print("msg2", len(msg))
                        # 接收服务器的flag信息，是否发送文件
                        flagsize = struct.calcsize(">I")
                        recv_msg = p.recv(flagsize)
                        flag = struct.unpack(">I", recv_msg)[0]
                        # print("flag", flag)
                        # if flag == -1:
                        #     continue
                        # 发送
                        if flag:
                            msg = struct.pack(f">{buffer_size}s", buffer)
                            # print("msg3", len(msg))
                            if buffer_size > SEND_BUF_SIZE:
                                send_index = 0

                                while send_index < buffer_size:
                                    send_size = min(SEND_BUF_SIZE, buffer_size - send_index)
                                    p.sendall(msg[send_index:send_index + send_size])
                                    send_index += send_size
                            else:
                                p.sendall(msg)
                            print(f"send file: {filepath}")
                        else:
                            print(f"pass file: {filepath}")

    p.close()


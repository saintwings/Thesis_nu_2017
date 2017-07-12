from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QMainWindow, QTextEdit,
    QAction, QFileDialog, QApplication)
from SetPostureHumanoidUI_QT5 import Ui_Form
import time
import serial
import sys
import serial.tools.list_ports

class HumanoidMainWindow(QtWidgets.QMainWindow,Ui_Form):
    int_id_L =[1,2,3,4,5,6]
    int_id_R =[11,12,13,14,15,16]
    int_id_LArm =[21,22,23,24]
    int_id_RArm =[31,32,33,34]
    int_id_H =[41,42,43]
    int_id_All = int_id_L + int_id_R + int_id_LArm + int_id_RArm + int_id_H
    int_motor_Amount = 23
    int_keyframe_Amount = 30
    int_time_Initial = 20

    def __init__(self, parent=None):
        super(HumanoidMainWindow, self).__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.InitVariable()
        self.InitUI()
        self.SetButtonAndSpinCtrlDisable()


    def InitVariable(self):
        self.int_stepTime = 0.03
        self.str_keyframeSelected ='Keyframe1'
        self.int_keyframeSelected = 1
        self.bool_comportConnected = False
        self.int_numberOfKeyframe = 0
        self.str_fileName = 'Unknown'
        self.str_comport = 'com81'
        self.str_baudrate = 1000000
        self.int_keyframe = 0
        self.int_motorID = 0
        self.bool_activeKeyframe =[False for x in range (self.int_keyframe_Amount)]
        file_center = open('motor_center.txt', 'r')
        self.int_motorCenterValue = file_center.read()
        file_center.close()
        self.int_motorCenterValue = self.int_motorCenterValue.split('\n')
        print (self.int_motorCenterValue)
        #cast motorCenterValue from str to int#
        for x in range (self.int_motor_Amount):
            self.int_motorCenterValue[x] = int(self.int_motorCenterValue[x])
        file_type = open('motor_type.txt', 'r')
        self.str_motorType = file_type.read()
        print(self.str_motorType)
        file_type.close()
        self.str_motorType = self.str_motorType.split('\n')
        print(self.str_motorType)
        #print self.int_motorCenterValue
        #print len(self.int_motorCenterValue)
        ##self.int_motorValue[keyframe0-14][motorID0-17]##
        self.int_old_motorValue = [self.int_motorCenterValue[x] for x in range (self.int_motor_Amount)]
        self.int_backup_motorValue = [self.int_motorCenterValue[x] for x in range (self.int_motor_Amount)]
        self.int_motorValue = [[self.int_motorCenterValue[x] for x in range (self.int_motor_Amount)] for y in range (self.int_keyframe_Amount)]

        self.dic_motorIndexID = {'id1':0,'id2':1,'id3':2,'id4':3,'id5':4,'id6':5,
                                 'id11':6,'id12':7,'id13':8,'id14':9,'id15':10,'id16':11,
                                 'id21':12,'id22':13,'id23':14,'id24':15,
                                 'id31':16,'id32':17,'id33':18,'id34':19,
                                 'id41':20,'id42':21,'id43':22}
        self.int_time = [self.int_time_Initial for x in range (self.int_keyframe_Amount)]

    def InitUI(self):

        self.SetMotorCenterLabel()

        baudrateList = ['9600','115200','1000000']
        self.ui.baudrate_comboBox.addItems(baudrateList)

        postureList = ['unknown','p1','p2','p3','p4','p5','p6','p7','p8','p9','p10']
        self.ui.posture_comboBox.addItems(postureList)

        self.keyframeList = [str(i) for i in range(1, 31)]

        self.ui.keyFrame_comboBox.addItems(self.keyframeList)

        self.ui.connectionStatus_label.setText("Status : Disconnect")

        self.ui.activeKeyframe_checkBox.clicked.connect(self.ActiveKeyframe_CheckBox)
        self.ui.keyFrame_comboBox.activated[str].connect(self.OnSelect_ComboboxKeyframe)
        self.ui.posture_comboBox.activated[str].connect(self.OnSelect_ComboboxPosture)
        self.ui.comport_comboBox.activated[str].connect(self.OnSelect_ComboboxComport)
        self.ui.baudrate_comboBox.activated[str].connect(self.OnSelect_ComboboxBaudrate)

        self.ui.comport_comboBox.currentIndexChanged[str].connect(self.OnIndexChange_ComboboxComport)

        self.ui.connect_Button.clicked.connect(self.OnButton_connect)
        self.ui.loadPosture_pushButton.clicked.connect(self.OnButton_Load)
        self.ui.savePosture_pushButton.clicked.connect(self.OnButton_Save)
        self.ui.setReady_Button.clicked.connect(self.OnButton_ready)
        self.ui.playAll_Button.clicked.connect(self.OnButton_playAll)
        self.ui.setTime_pushButton.clicked.connect(self.OnButton_time)
        self.ui.play_pushButton.clicked.connect(self.OnButton_play)

        self.ui.setAll_pushButton.clicked.connect(self.OnButton_setAll)
        self.ui.setLAll_pushButton.clicked.connect(self.OnButton_setLAll)
        self.ui.setRAll_pushButton.clicked.connect(self.OnButton_setRAll)
        self.ui.setLArmAll_pushButton.clicked.connect(self.OnButton_setLArmAll)
        self.ui.setRArmAll_pushButton.clicked.connect(self.OnButton_setRArmAll)
        self.ui.setHAll_pushButton.clicked.connect(self.OnButton_setHAll)

        for id in self.int_id_All:
            eval("self.ui.motor{}Set_pushButton".format(id)).clicked.connect(lambda ignore, id=id: self.OnButton_Set(id))

            #QtCore.QObject.connect(getattr(self.ui,'motor'+str(i)+'Set_pushButton'),QtCore.SIGNAL("clicked()"), getattr(self,'OnButton_id'+str(i)+'Set'))
        ###### QtCore.QObject.connect(self.ui.motor1Set_pushButton,QtCore.SIGNAL("clicked()"), self.OnButton_id1Set)


        self.ui.getAll_pushButton.clicked.connect(self.OnButton_getAll)
        self.ui.getLAll_pushButton.clicked.connect(self.OnButton_getLAll)
        self.ui.getRAll_pushButton.clicked.connect(self.OnButton_getRAll)
        self.ui.getLArmAll_pushButton.clicked.connect(self.OnButton_getLArmAll)
        self.ui.getRArmAll_pushButton.clicked.connect(self.OnButton_getRArmAll)
        self.ui.getHAll_pushButton.clicked.connect(self.OnButton_getHAll)

        for id in self.int_id_All:
            eval("self.ui.motor{}Get_pushButton".format(id)).clicked.connect(
                lambda ignore, id=id: self.OnButton_Get(id))

            #QtCore.QObject.connect(getattr(self.ui,'motor'+str(i)+'Get_pushButton'),QtCore.SIGNAL("clicked()"), getattr(self,'OnButton_id'+str(i)+'Get'))


        self.ui.disTAll_pushButton.clicked.connect(self.OnButton_DisableTorqueAll)
        self.ui.disTLAll_pushButton.clicked.connect(self.OnButton_DisableTorqueLAll)
        self.ui.disTRAll_pushButton.clicked.connect(self.OnButton_DisableTorqueRAll)
        self.ui.disTLArmAll_pushButton.clicked.connect(self.OnButton_DisableTorqueLArmAll)
        self.ui.disTRArmAll_pushButton.clicked.connect(self.OnButton_DisableTorqueRArmAll)
        self.ui.disTHAll_pushButton.clicked.connect(self.OnButton_DisableTorqueHAll)

        ######QtCore.QObject.connect(self.ui.motor1DisT_pushButton,QtCore.SIGNAL("clicked()"), self.OnButton_id1DisableTorque)
        for id in self.int_id_All:
            eval("self.ui.motor{}DisT_pushButton".format(id)).clicked.connect(
                lambda ignore, id=id: self.OnButton_DisableTorque(id))

            #QtCore.QObject.connect(getattr(self.ui,'motor'+str(i)+'DisT_pushButton'),QtCore.SIGNAL("clicked()"), getattr(self,'OnButton_id'+str(i)+'DisableTorque'))

        self.ui.saveCenter_pushButton.clicked.connect(self.OnButton_SaveCenter)

        self.Search_Comport()

    def Search_Comport(self):
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            self.ui.comport_comboBox.addItem(p[0])

    def OnIndexChange_ComboboxComport(self,text):
        self.str_comport = str(text)
        print(self.str_fileName)

    def OnButton_Delete(self):
        #self.ui.keyFrame_comboBox.
        #self.int_backup_motorValue
        #self.int_keyframeSelected
        pass

    def OnButton_DisableTorqueAll(self):
        for i in self.int_id_All:
            self.setDisableMotorTorque(i)
            time.sleep(0.015)

    def OnButton_DisableTorqueLAll(self):
        for i in self.int_id_L:
            self.setDisableMotorTorque(i)
            time.sleep(0.015)

    def OnButton_DisableTorqueRAll(self):
        for i in self.int_id_R:
            self.setDisableMotorTorque(i)
            time.sleep(0.015)

    def OnButton_DisableTorqueLArmAll(self):
        for i in self.int_id_LArm:
            self.setDisableMotorTorque(i)
            time.sleep(0.015)

    def OnButton_DisableTorqueRArmAll(self):
        for i in self.int_id_RArm:
            self.setDisableMotorTorque(i)
            time.sleep(0.015)

    def OnButton_DisableTorqueHAll(self):
        for i in self.int_id_H:
            self.setDisableMotorTorque(i)
        time.sleep(0.015)

    def OnButton_DisableTorque(self, id):
        print("DisableTorque ID " + str(id))
        self.setDisableMotorTorque(id)

    def OnButton_Get(self, id):
        print("get ID = " + str(id))
        eval("self.ui.motor{}Value_spinBox.setValue(self.getMotorPosition(id))".format(id))

    def OnButton_getAll(self):
        print("getAll")
        for id in self.int_id_All:
            eval("self.ui.motor{}Value_spinBox.setValue(self.getMotorPosition(id))".format(id))
            time.sleep(0.015)

    def OnButton_getLAll(self):
        print("get_L_All")
        for id in self.int_id_L:
            eval("self.ui.motor{}Value_spinBox.setValue(self.getMotorPosition(id))".format(id))
            time.sleep(0.015)

    def OnButton_getRAll(self):
        print("get_R_All")
        for id in self.int_id_R:
            eval("self.ui.motor{}Value_spinBox.setValue(self.getMotorPosition(id))".format(id))
            time.sleep(0.015)

    def OnButton_getLArmAll(self):
        print("get_L_Arm_All")
        for id in self.int_id_LArm:
            eval("self.ui.motor{}Value_spinBox.setValue(self.getMotorPosition(id))".format(id))
            time.sleep(0.015)

    def OnButton_getRArmAll(self):
        print("get_R_Arm_All")
        for id in self.int_id_RArm:
            eval("self.ui.motor{}Value_spinBox.setValue(self.getMotorPosition(id))".format(id))
            time.sleep(0.015)

    def OnButton_getHAll(self):
        print("get_H_All")
        for id in self.int_id_H:
            eval("self.ui.motor{}Value_spinBox.setValue(self.getMotorPosition(id))".format(id))
            time.sleep(0.015)

    def OnButton_Set(self, id):
        print("set id=",id)
        self.int_motorValue[self.GetOrderKeyframe() - 1][self.dic_motorIndexID['id' + str(id)]] = eval(
            "self.ui.motor{}Value_spinBox.value()".format(id))
        self.setDeviceMoving(self.str_comport, self.str_baudrate, id, "Ex",
                             self.int_motorValue[self.GetOrderKeyframe() - 1][self.dic_motorIndexID['id' + str(id)]],
                             1023, 1023)
        self.int_old_motorValue[self.dic_motorIndexID['id' + str(id)]] = \
        self.int_motorValue[self.GetOrderKeyframe() - 1][self.dic_motorIndexID['id' + str(id)]]

    def OnButton_play(self):
        print("play...")

        self.SetButtonAndSpinCtrlDisable()


        for id in self.int_id_All:
            self.int_motorValue[self.GetOrderKeyframe() - 1][eval("self.dic_motorIndexID['id{}']".format(id))] = eval(
                "self.ui.motor{}Value_spinBox.value()".format(id))

        time_start = time.time()
        time_finish = time_start + float(self.int_time[self.GetOrderKeyframe() - 1])/10
        in_time = True

        print(time_start)
        print(time_finish)
        print('Wait....')
        while in_time:
            time_current = time.time()
            if time_current >= time_finish:

                for i in self.int_id_All:
                    self.setDeviceMoving( self.str_comport, self.str_baudrate, i, "Ex", self.int_motorValue[self.GetOrderKeyframe() - 1][self.dic_motorIndexID['id'+str(i)]], 1023, 1023)
                    #self.setDeviceMoving( self.str_comport, self.str_baudrate, 1, "Ex", self.int_motorValue[self.GetOrderKeyframe() - 1][self.dic_motorIndexID['id1']], 1023, 1023)

                for id in self.int_id_All:
                    self.int_old_motorValue[eval("self.dic_motorIndexID['id{}']".format(id))] = \
                    self.int_motorValue[self.GetOrderKeyframe() - 1][eval("self.dic_motorIndexID['id{}']".format(id))]

                in_time = False

            else:
                for i in self.int_id_All:
                    self.setDeviceMoving( self.str_comport, self.str_baudrate, i, "Ex", self.InterpolateMotorValue(self.int_motorValue[self.GetOrderKeyframe() - 1][self.dic_motorIndexID['id'+str(i)]],self.int_old_motorValue[self.dic_motorIndexID['id'+str(i)]],time_finish,time_start,time_current), 1023, 1023)
                    #self.setDeviceMoving( self.str_comport, self.str_baudrate, 1, "Ex", self.InterpolateMotorValue(self.int_motorValue[self.GetOrderKeyframe() - 1][self.dic_motorIndexID['id1']],self.int_old_motorValue[self.dic_motorIndexID['id1']],time_finish,time_start,time_current), 1023, 1023)

            time.sleep(self.int_stepTime)

        print('Finished')
        self.SetButtonAndSpinCtrlEnable()

    def OnButton_setLAll(self):
        print("set L all")
        for id in self.int_id_L:
            self.int_motorValue[self.GetOrderKeyframe() - 1][eval("self.dic_motorIndexID['id{}']".format(id))] = eval(
                "self.ui.motor{}Value_spinBox.value()".format(id))
            self.setDeviceMoving(self.str_comport, self.str_baudrate, id, "Ex",
                                 self.int_motorValue[self.GetOrderKeyframe() - 1][
                                     eval("self.dic_motorIndexID['id{}']".format(id))], 1023, 1023)
            self.int_old_motorValue[eval("self.dic_motorIndexID['id{}']".format(id))] = \
            self.int_motorValue[self.GetOrderKeyframe() - 1][eval("self.dic_motorIndexID['id{}']".format(id))]
            time.sleep(0.015)
    def OnButton_setRAll(self):
        print("set R all")
        for id in self.int_id_R:
            self.int_motorValue[self.GetOrderKeyframe() - 1][eval("self.dic_motorIndexID['id{}']".format(id))] = eval(
                "self.ui.motor{}Value_spinBox.value()".format(id))
            self.setDeviceMoving(self.str_comport, self.str_baudrate, id, "Ex",
                                 self.int_motorValue[self.GetOrderKeyframe() - 1][
                                     eval("self.dic_motorIndexID['id{}']".format(id))], 1023, 1023)
            self.int_old_motorValue[eval("self.dic_motorIndexID['id{}']".format(id))] = \
            self.int_motorValue[self.GetOrderKeyframe() - 1][eval("self.dic_motorIndexID['id{}']".format(id))]
            time.sleep(0.015)
    def OnButton_setLArmAll(self):
        print("set L arm all")
        for id in self.int_id_LArm:
            self.int_motorValue[self.GetOrderKeyframe() - 1][eval("self.dic_motorIndexID['id{}']".format(id))] = eval(
                "self.ui.motor{}Value_spinBox.value()".format(id))
            self.setDeviceMoving(self.str_comport, self.str_baudrate, id, "Ex",
                                 self.int_motorValue[self.GetOrderKeyframe() - 1][
                                     eval("self.dic_motorIndexID['id{}']".format(id))], 1023, 1023)
            self.int_old_motorValue[eval("self.dic_motorIndexID['id{}']".format(id))] = \
            self.int_motorValue[self.GetOrderKeyframe() - 1][eval("self.dic_motorIndexID['id{}']".format(id))]
            time.sleep(0.015)
    def OnButton_setRArmAll(self):
        print("set R arm all")
        for id in self.int_id_RArm:
            self.int_motorValue[self.GetOrderKeyframe() - 1][eval("self.dic_motorIndexID['id{}']".format(id))] = eval(
                "self.ui.motor{}Value_spinBox.value()".format(id))
            self.setDeviceMoving(self.str_comport, self.str_baudrate, id, "Ex",
                                 self.int_motorValue[self.GetOrderKeyframe() - 1][
                                     eval("self.dic_motorIndexID['id{}']".format(id))], 1023, 1023)
            self.int_old_motorValue[eval("self.dic_motorIndexID['id{}']".format(id))] = \
            self.int_motorValue[self.GetOrderKeyframe() - 1][eval("self.dic_motorIndexID['id{}']".format(id))]
            time.sleep(0.015)
    def OnButton_setHAll(self):
        print("set H all")
        for id in self.int_id_H:
            self.int_motorValue[self.GetOrderKeyframe() - 1][eval("self.dic_motorIndexID['id{}']".format(id))] = eval(
                "self.ui.motor{}Value_spinBox.value()".format(id))
            self.setDeviceMoving(self.str_comport, self.str_baudrate, id, "Ex",
                                 self.int_motorValue[self.GetOrderKeyframe() - 1][
                                     eval("self.dic_motorIndexID['id{}']".format(id))], 1023, 1023)
            self.int_old_motorValue[eval("self.dic_motorIndexID['id{}']".format(id))] = \
            self.int_motorValue[self.GetOrderKeyframe() - 1][eval("self.dic_motorIndexID['id{}']".format(id))]
            time.sleep(0.015)
    def OnButton_setAll(self):
        print("set all")
        #self.int_time[self.GetOrderKeyframe() - 1] = self.spinctrl_time.GetValue()
        for id in self.int_id_All:
            self.int_motorValue[self.GetOrderKeyframe() - 1][eval("self.dic_motorIndexID['id{}']".format(id))] = eval(
                "self.ui.motor{}Value_spinBox.value()".format(id))
            self.setDeviceMoving(self.str_comport, self.str_baudrate, id, "Ex",
                                 self.int_motorValue[self.GetOrderKeyframe() - 1][
                                     eval("self.dic_motorIndexID['id{}']".format(id))], 1023, 1023)
            self.int_old_motorValue[eval("self.dic_motorIndexID['id{}']".format(id))] = \
            self.int_motorValue[self.GetOrderKeyframe() - 1][eval("self.dic_motorIndexID['id{}']".format(id))]
            time.sleep(0.015)
    def OnButton_time(self):
        self.int_time[self.GetOrderKeyframe() - 1] = self.ui.keyframeTime_spinBox.value()
        print(self.int_time[self.GetOrderKeyframe() - 1])

    def OnButton_ready(self):
        print("ready...")

        self.SetButtonAndSpinCtrlDisable()

        if self.int_numberOfKeyframe == 0:
            print('Error!! Number of keyframe = 0 ')
        else:
            time_start = time.time()
            time_finish = time_start + float(self.int_time[0])/10
            in_time = True

            print(time_start)
            print(time_finish)
            print('Wait....')
            while in_time:
                time_current = time.time()
                if time_current >= time_finish:
                    for i in self.int_id_All:
                        self.setDeviceMoving( self.str_comport, self.str_baudrate, i, "Ex", self.int_motorValue[0][self.dic_motorIndexID['id'+str(i)]], 200, 200)
                        self.int_old_motorValue[eval("self.dic_motorIndexID['id{}']".format(i))] = \
                        self.int_motorValue[0][eval("self.dic_motorIndexID['id{}']".format(i))]

                    in_time = False

                else:
                    for i in self.int_id_All:
                        self.setDeviceMoving( self.str_comport, self.str_baudrate, i, "Ex", self.InterpolateMotorValue(self.int_motorValue[0][self.dic_motorIndexID['id'+str(i)]],self.int_old_motorValue[self.dic_motorIndexID['id'+str(i)]],time_finish,time_start,time_current), 200, 200)
                        #self.setDeviceMoving( self.str_comport, self.str_baudrate, 1, "Ex", self.InterpolateMotorValue(self.int_motorValue[0][self.dic_motorIndexID['id1']],self.int_old_motorValue[self.dic_motorIndexID['id1']],time_finish,time_start,time_current), 200, 200)

                time.sleep(0.015)
            print('Finished')
        self.SetButtonAndSpinCtrlEnable()

    def OnButton_playAll(self):
        print("play all")
        self.SetButtonAndSpinCtrlDisable()

        if self.int_numberOfKeyframe == 0:
            print('Error!! Number of keyframe = 0 ')
        else:
            self.SetButtonAndSpinCtrlDisable()
            for x in range(self.int_numberOfKeyframe):
                time_start = time.time()
                time_finish = time_start + float(self.int_time[x])/10
                in_time = True

                print(time_start)
                print(time_finish)
                print('keyframe = ' + str(x + 1))
                print('Time = ' + str(self.int_time[x]))
                print('Wait....')
                while in_time:
                    time_current = time.time()
                    if time_current >= time_finish:
                        for i in self.int_id_All:
                            self.setDeviceMoving( self.str_comport, self.str_baudrate, i, "Ex", self.int_motorValue[x][self.dic_motorIndexID['id'+str(i)]], 1023, 1023)
                            self.int_old_motorValue[eval("self.dic_motorIndexID['id{}']".format(i))] = \
                            self.int_motorValue[x][eval("self.dic_motorIndexID['id{}']".format(i))]

                        in_time = False

                    else:
                        for i in self.int_id_All:
                            self.setDeviceMoving( self.str_comport, self.str_baudrate, i, "Ex", self.InterpolateMotorValue(self.int_motorValue[x][self.dic_motorIndexID['id'+str(i)]],self.int_old_motorValue[self.dic_motorIndexID['id'+str(i)]],time_finish,time_start,time_current), 1023, 1023)
                            #self.setDeviceMoving( self.str_comport, self.str_baudrate, 1, "Ex", self.InterpolateMotorValue(self.int_motorValue[x][self.dic_motorIndexID['id1']],self.int_old_motorValue[self.dic_motorIndexID['id1']],time_finish,time_start,time_current), 1023, 1023)

                    time.sleep(self.int_stepTime)

                print('Finished')
            self.SetButtonAndSpinCtrlEnable()

    def OnButton_Load(self):
        print("Load")
        print(self.str_fileName)

        self.ui.fileName_label.setText(self.str_fileName)

        namePosture = self.str_fileName + '.txt'
        print(namePosture)

        file_posture = open(namePosture, 'r')
        str_load_data = file_posture.read()
        file_posture.close()
        str_load_data = str_load_data.split('\n')
        self.int_numberOfKeyframe = int(str_load_data[0])

        #self.text_atSub0_numberOfKeyframe.SetLabel(str(self.int_numberOfKeyframe))
        self.ui.numOfKeyframeStatus_label.setText(str(self.int_numberOfKeyframe))

        int_count_data = 1
        for x in range (self.int_numberOfKeyframe):
            self.bool_activeKeyframe[x] = True
            for y in range (self.int_motor_Amount):
                self.int_motorValue[x][y] = int(str_load_data[int_count_data])
                int_count_data = int_count_data + 1
        for z in range (self.int_numberOfKeyframe,self.int_keyframe_Amount):
            self.bool_activeKeyframe[z] = False

        nameTime = self.str_fileName + '_time.txt'

        file_Time = open(nameTime,'r')
        str_load_data = file_Time.read()
        file_Time.close()
        str_load_data = str_load_data.split('\n')
        int_count_data = 1
        for x in range (self.int_numberOfKeyframe):
            self.int_time[x] = int(str_load_data[int_count_data])
            int_count_data = int_count_data + 1

        self.SetValueKeyframeToShow()

    def OnButton_Save(self):
        print("Save")
        print(self.str_fileName)

        self.ui.fileName_label.setText(self.str_fileName)

        namePosture = self.str_fileName + '.txt'
        print(namePosture)
        file_posture = open(namePosture, 'w')
        file_posture.write(str(self.int_numberOfKeyframe)+'\n')
        for x in range (self.int_numberOfKeyframe):
            #file_frontGetup.write(str(x+1)+'\n')
            for y in range (self.int_motor_Amount):
                file_posture.write(str(self.int_motorValue[x][y])+'\n')

        file_posture.close()

        namePosture = self.str_fileName + '_3.txt'
        file_posture = open(namePosture, 'w')
        file_posture.write(str(self.int_numberOfKeyframe)+'\n')
        for x in range (self.int_numberOfKeyframe):
            #file_frontGetup.write(str(x+1)+'\n')
            file_posture.write('{ ')
            for y in range (self.int_motor_Amount-4):
                if y != 15:
                    file_posture.write(str(self.int_motorValue[x][y])+' ,')
            file_posture.write('/*Time*/ ')
            file_posture.write(str(self.int_time[x])+'*0.1}\n')

        file_posture.close()

        nameTime = self.str_fileName + '_time.txt'

        file_Time = open(nameTime,'w')
        file_Time.write(str(self.int_numberOfKeyframe)+'\n')
        for x in range (self.int_numberOfKeyframe):
            file_Time.write(str(self.int_time[x])+'\n')
        file_Time.close()

    def SetMotorCenterLabel(self):
        for i in self.int_id_All:
            eval(
                "self.ui.motor{}center_label.setText(str(self.int_motorCenterValue[self.dic_motorIndexID['id'+str(i)]]))".format(
                    i))

    def OnButton_SaveCenter(self):
        file_center = open('motor_center.txt', 'w')
        for i in self.int_id_All:
            self.int_motorCenterValue[self.dic_motorIndexID['id'+str(i)]] = getattr(self.ui,"motor"+str(i)+"Value_spinBox").value()
            #self.int_motorCenterValue[self.dic_motorIndexID['id1']] = self.ui.motor1Value_spinBox.value()

        for y in range (self.int_motor_Amount):
                file_center.write(str(self.int_motorCenterValue[y])+'\n')

        file_center.close()

        self.SetMotorCenterLabel()

    def OnSelect_ComboboxPosture(self,text):
        self.str_fileName = text
        print(self.str_fileName)

    def OnButton_connect(self):
        print("connect clicked")
        if self.bool_comportConnected == False:
            self.bool_comportConnected = True
            self.serialDevice = serial.Serial(self.str_comport, self.str_baudrate,8,'N',1,0,0,0,0)
            self.ui.connectionStatus_label.setText("Status : Connected")
            self.ui.connect_Button.setText("Disconnect")
        else:
            self.bool_comportConnected = False
            self.serialDevice.close()
            self.ui.connectionStatus_label.setText("Status : Disconnected")
            self.ui.connect_Button.setText("Connect")

    def OnSelect_ComboboxComport(self,text):
        self.str_comport = str(text)
        print(self.str_comport)

    def OnSelect_ComboboxBaudrate(self,text):
        self.str_baudrate = str(text)
        print(self.str_baudrate)

    def OnSelect_ComboboxKeyframe(self,text):
        self.str_keyframeSelected = text
        self.int_keyframeSelected = int(text)
        print(self.int_keyframeSelected)
        self.SetValueKeyframeToShow()

    def SetValueKeyframeToShow(self):

        keyframe = self.int_keyframeSelected

        self.int_keyframeSelected = keyframe

        print("keyframe selected = ")
        print(self.int_keyframeSelected)


        if self.bool_activeKeyframe[keyframe-1] == True:
            self.ui.activeKeyframe_checkBox.setChecked(2)
            self.SetButtonAndSpinCtrlEnable()

            for id in self.int_id_All:
                eval("self.ui.motor{}Value_spinBox".format(id)).setValue(
                    self.int_motorValue[keyframe - 1][eval("self.dic_motorIndexID['id{}']".format(id))])


            self.ui.keyframeTime_spinBox.setValue(self.int_time[keyframe-1])
        else:
            self.ui.activeKeyframe_checkBox.setChecked(0)
            self.SetButtonAndSpinCtrlDisable()

    def CheckPreviousKeyframe(self,currentKeyframe):
        if currentKeyframe == 1:
            self.bool_activeKeyframe[currentKeyframe-1] = True
            self.SetValueKeyframeToShow()
        else:
            self.bool_activeKeyframe[0] = True
            bool_getActiveKeyframe = False
            int_searchKeyframe = currentKeyframe - 1
            while(bool_getActiveKeyframe == False):
                if self.bool_activeKeyframe[int_searchKeyframe - 1] == True:
                    bool_getActiveKeyframe = True
                else:
                    int_searchKeyframe = int_searchKeyframe - 1
            for i in range (int_searchKeyframe+1,currentKeyframe+1):
                self.bool_activeKeyframe[i-1] = True
                for j in range (self.int_motor_Amount):
                    self.int_motorValue[i-1][j] = self.int_motorValue[int_searchKeyframe-1][j]
            #self.SetValueKeyframeToShow(currentKeyframe)
            self.SetValueKeyframeToShow()

    def CheckNextKeyframe(self,currentKeyframe):
        if currentKeyframe == self.int_keyframe_Amount:
            self.bool_activeKeyframe[currentKeyframe-1] = False
            self.SetValueKeyframeToShow()
        else:
            self.bool_activeKeyframe[self.int_keyframe_Amount-1] = False
            bool_getNotActiveKeyframe = False
            int_searchKeyframe = currentKeyframe + 1
            while(bool_getNotActiveKeyframe == False):
                if self.bool_activeKeyframe[int_searchKeyframe - 1] == False:
                    bool_getNotActiveKeyframe = True
                else:
                    int_searchKeyframe = int_searchKeyframe + 1
            for i in range (currentKeyframe,int_searchKeyframe+1):
                self.bool_activeKeyframe[i-1] = False
                for j in range (self.int_motor_Amount):
                    self.int_motorValue[i-1][j] = self.int_motorCenterValue[j]
            #self.SetValueKeyframeToShow(currentKeyframe)
            self.SetValueKeyframeToShow()

    def ActiveKeyframe_CheckBox(self):
        print(self.ui.activeKeyframe_checkBox.checkState())

        if self.ui.activeKeyframe_checkBox.checkState() == 2:
            print("Checked")

            self.CheckPreviousKeyframe(self.int_keyframeSelected)
            self.int_numberOfKeyframe = self.int_keyframeSelected


            self.ui.numOfKeyframeStatus_label.setText(str(self.int_numberOfKeyframe))

        else:
            print("Unchecked")
            self.CheckNextKeyframe(self.int_keyframeSelected)
            self.int_numberOfKeyframe = (self.int_keyframeSelected - 1)

            self.ui.numOfKeyframeStatus_label.setText(str(self.int_numberOfKeyframe))

    def SetButtonAndSpinCtrlEnable(self):

        self.ui.setAll_pushButton.setEnabled(True)
        self.ui.setLAll_pushButton.setEnabled(True)
        self.ui.setRAll_pushButton.setEnabled(True)
        self.ui.setLArmAll_pushButton.setEnabled(True)
        self.ui.setRArmAll_pushButton.setEnabled(True)
        self.ui.setHAll_pushButton.setEnabled(True)

        self.ui.getAll_pushButton.setEnabled(True)
        self.ui.getLAll_pushButton.setEnabled(True)
        self.ui.getRAll_pushButton.setEnabled(True)
        self.ui.getLArmAll_pushButton.setEnabled(True)
        self.ui.getRArmAll_pushButton.setEnabled(True)
        self.ui.getHAll_pushButton.setEnabled(True)

        self.ui.disTAll_pushButton.setEnabled(True)
        self.ui.disTLAll_pushButton.setEnabled(True)
        self.ui.disTRAll_pushButton.setEnabled(True)
        self.ui.disTLArmAll_pushButton.setEnabled(True)
        self.ui.disTRArmAll_pushButton.setEnabled(True)
        self.ui.disTHAll_pushButton.setEnabled(True)

        self.ui.deleteKeyframe_pushButton.setEnabled(True)
        self.ui.duplicateKeyframe_pushButton.setEnabled(True)
        self.ui.previousSwitchKeyframe_pushButton.setEnabled(True)
        self.ui.nextSwitchKeyframe_pushButton.setEnabled(True)
        self.ui.play_pushButton.setEnabled(True)
        self.ui.playAll_Button.setEnabled(True)
        self.ui.setReady_Button.setEnabled(True)
        self.ui.setTime_pushButton.setEnabled(True)
        self.ui.setAll_pushButton.setEnabled(True)
        self.ui.keyframeTime_spinBox.setEnabled(True)

        for id in self.int_id_All:
            eval("self.ui.motor{}Value_spinBox.setEnabled(True)".format(id))
            eval("self.ui.motor{}value_dial.setEnabled(True)".format(id))
            eval("self.ui.motor{}Set_pushButton.setEnabled(True)".format(id))
            eval("self.ui.motor{}Get_pushButton.setEnabled(True)".format(id))
            eval("self.ui.motor{}DisT_pushButton.setEnabled(True)".format(id))

    def SetButtonAndSpinCtrlDisable(self):

        self.ui.setAll_pushButton.setDisabled(True)
        self.ui.setLAll_pushButton.setDisabled(True)
        self.ui.setRAll_pushButton.setDisabled(True)
        self.ui.setLArmAll_pushButton.setDisabled(True)
        self.ui.setRArmAll_pushButton.setDisabled(True)
        self.ui.setHAll_pushButton.setDisabled(True)

        self.ui.getAll_pushButton.setDisabled(True)
        self.ui.getLAll_pushButton.setDisabled(True)
        self.ui.getRAll_pushButton.setDisabled(True)
        self.ui.getLArmAll_pushButton.setDisabled(True)
        self.ui.getRArmAll_pushButton.setDisabled(True)
        self.ui.getHAll_pushButton.setDisabled(True)

        self.ui.disTAll_pushButton.setDisabled(True)
        self.ui.disTLAll_pushButton.setDisabled(True)
        self.ui.disTRAll_pushButton.setDisabled(True)
        self.ui.disTLArmAll_pushButton.setDisabled(True)
        self.ui.disTRArmAll_pushButton.setDisabled(True)
        self.ui.disTHAll_pushButton.setDisabled(True)

        self.ui.deleteKeyframe_pushButton.setDisabled(True)
        self.ui.duplicateKeyframe_pushButton.setDisabled(True)
        self.ui.previousSwitchKeyframe_pushButton.setDisabled(True)
        self.ui.nextSwitchKeyframe_pushButton.setDisabled(True)
        self.ui.play_pushButton.setDisabled(True)
        self.ui.playAll_Button.setDisabled(True)
        self.ui.setReady_Button.setDisabled(True)
        self.ui.setTime_pushButton.setDisabled(True)
        self.ui.setAll_pushButton.setDisabled(True)
        self.ui.keyframeTime_spinBox.setDisabled(True)

        for id in self.int_id_All:
            eval("self.ui.motor{}Value_spinBox.setDisabled(True)".format(id))
            eval("self.ui.motor{}value_dial.setDisabled(True)".format(id))
            eval("self.ui.motor{}Set_pushButton.setDisabled(True)".format(id))
            eval("self.ui.motor{}Get_pushButton.setDisabled(True)".format(id))
            eval("self.ui.motor{}DisT_pushButton.setDisabled(True)".format(id))

    def GetOrderKeyframe(self):

        for index, kf in enumerate(self.keyframeList):
            if self.int_keyframeSelected == int(kf):
                orderKeyframe = index + 1
        return orderKeyframe

    def setReadMotorPacket(self,deviceID,Offset,Length):
        readPacket = [0xFF, 0xFF, deviceID, 0x04, 0x02, Offset, Length]
        checkSumOrdList = readPacket[2:]
        checkSumOrdListSum = sum(checkSumOrdList)
        computedCheckSum = (~(checkSumOrdListSum % 256)) % 256
        readPacket.append(computedCheckSum)
        self.serialDevice.write(readPacket)
        print(readPacket)

    def getMotorQueryResponse( self, deviceID, Length ):

            queryData = 0
            responsePacketSize = Length + 6
            #responsePacket = readAllData(serialDevice)
            responsePacket = self.serialDevice.read(self.serialDevice.inWaiting())



            if len(responsePacket) == responsePacketSize:

                print("responsePacket=", responsePacket)

                responseID = responsePacket[2]
                errorByte = responsePacket[4]

                ### python 3
                if responseID == deviceID and errorByte == 0:
                    if Length == 2:
                        queryData = responsePacket[5] + 256 * responsePacket[6]
                    elif Length == 1:
                        queryData = responsePacket[5]
                        # print "Return data:", queryData
                else:
                    print("Error response:", responseID, errorByte)

                responsePacketStatue = True

            else:
                responsePacketStatue = False

            print("queryData=", queryData)
            return queryData,responsePacketStatue

    def get(self,deviceID, address, Length):

            for i in range(0,5):
                self.setReadMotorPacket(deviceID, address, Length)
                time.sleep(0.02)
                data, status = self.getMotorQueryResponse(deviceID, Length)

                if status == True:
                    break
                else:
                    print("motor ID " + str(deviceID) + "  no response " + str(i))

            return data

    def getMotorPosition(self,id):
            data = self.get(id,36,2)
            return data

    def rxPacketConversion( self,value ):
            if value < 1024 and value >= 0:
                    hiByte = int(value/256)
                    loByte = value%256
            else:
                    print("rxPacketConversion: value out of range", value)
            return loByte, hiByte

    def exPacketConversion( self,value ):
            if value < 4096 and value >= 0:
                    hiByte = int(value/256)
                    loByte = value%256
            else:
                    print("exPacketConversion: value out of range", value)
            return loByte, hiByte

    def setDisableMotorTorque(self,deviceID):
        Offset = 0x18
        Packet = [0xFF, 0xFF, deviceID, 0x04, 0x03, Offset, 0x00]
        checkSumOrdList = Packet[2:]
        checkSumOrdListSum = sum(checkSumOrdList)
        computedCheckSum = (~(checkSumOrdListSum % 256)) % 256
        Packet.append(computedCheckSum)
        self.serialDevice.write(Packet)
        print(Packet)

    def setDeviceMoving( self,Port, Baud, deviceID, deviceType, goalPos, goalSpeed, maxTorque):

        Offset = 0x1E
        Length = 6
        numberOfServo = 1
        packetLength = int((6 + 1) * numberOfServo + 4)
        (goalSpeedL, goalSpeedH) = self.rxPacketConversion(goalSpeed)
        (maxTorqueL, maxTorqueH) = self.rxPacketConversion(maxTorque)

        syncWritePacket = [0xFF, 0xFF, 0xFE, packetLength, 0x83, Offset, Length]
        if deviceType == "Rx" or deviceType == "Ax":
            (positionL, positionH) = self.rxPacketConversion(goalPos)
        elif deviceType == "Ex" or deviceType == "Mx":
            (positionL, positionH) = self.exPacketConversion(goalPos)
        parameterList = [deviceID, positionL, positionH, goalSpeedL, goalSpeedH, maxTorqueL, maxTorqueH]
        for parameter in parameterList:
            syncWritePacket.append(parameter)

        checkSumOrdList = syncWritePacket[2:]
        checkSumOrdListSum = sum(checkSumOrdList)
        computedCheckSum = (~(checkSumOrdListSum % 256)) % 256
        syncWritePacket.append(computedCheckSum)
        self.serialDevice.write(syncWritePacket)


        # print(syncWritePacket,"goalPos =",goalPos)
    def InterpolateMotorValue(self,finish_value,start_value,finish_time,start_time,current_time):
        motor_value = int((finish_value - start_value)*(current_time-start_time)/(finish_time - start_time)+start_value)
        #print motor_value
        return motor_value

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = HumanoidMainWindow()


    MainWindow.show()
    sys.exit(app.exec_())
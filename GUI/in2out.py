# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'in2out.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1032, 786)
        Dialog.setMinimumSize(QtCore.QSize(60, 0))
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(660, 730, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.load = QtWidgets.QPushButton(Dialog)
        self.load.setGeometry(QtCore.QRect(50, 40, 89, 25))
        self.load.setObjectName("load")
        self.save = QtWidgets.QPushButton(Dialog)
        self.save.setGeometry(QtCore.QRect(210, 40, 89, 25))
        self.save.setObjectName("save")
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(30, 90, 971, 314))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.input_label = QtWidgets.QLabel(self.verticalLayoutWidget_5)
        self.input_label.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.input_label.sizePolicy().hasHeightForWidth())
        self.input_label.setSizePolicy(sizePolicy)
        self.input_label.setMinimumSize(QtCore.QSize(200, 200))
        self.input_label.setMaximumSize(QtCore.QSize(200, 200))
        self.input_label.setText("")
        self.input_label.setPixmap(QtGui.QPixmap("autoenc.png"))
        self.input_label.setObjectName("input_label")
        self.horizontalLayout.addWidget(self.input_label)
        spacerItem1 = QtWidgets.QSpacerItem(88, 236, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.model_label = QtWidgets.QLabel(self.verticalLayoutWidget_5)
        self.model_label.setMinimumSize(QtCore.QSize(300, 200))
        self.model_label.setMaximumSize(QtCore.QSize(300, 200))
        self.model_label.setText("")
        self.model_label.setPixmap(QtGui.QPixmap("GUI/images/dl_model.jpg"))
        self.model_label.setScaledContents(True)
        self.model_label.setObjectName("model_label")
        self.verticalLayout.addWidget(self.model_label)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem3)
        self.horizontalLayout.addLayout(self.verticalLayout)
        spacerItem4 = QtWidgets.QSpacerItem(88, 236, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem4)
        self.output_label = QtWidgets.QLabel(self.verticalLayoutWidget_5)
        self.output_label.setMinimumSize(QtCore.QSize(200, 200))
        self.output_label.setMaximumSize(QtCore.QSize(200, 200))
        self.output_label.setText("")
        self.output_label.setPixmap(QtGui.QPixmap("autoenc.png"))
        self.output_label.setObjectName("output_label")
        self.horizontalLayout.addWidget(self.output_label)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem5)
        self.verticalLayout_7.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem6 = QtWidgets.QSpacerItem(25, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem6)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.back = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        self.back.setObjectName("back")
        self.verticalLayout_2.addWidget(self.back)
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem7)
        self.horizontalLayout_4.addLayout(self.verticalLayout_2)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem8)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.next = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        self.next.setObjectName("next")
        self.verticalLayout_3.addWidget(self.next)
        spacerItem9 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem9)
        self.horizontalLayout_4.addLayout(self.verticalLayout_3)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_4)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem10)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.progressBar = QtWidgets.QProgressBar(self.verticalLayoutWidget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(200)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy)
        self.progressBar.setMinimumSize(QtCore.QSize(200, 0))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_4.addWidget(self.progressBar)
        spacerItem11 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem11)
        self.horizontalLayout_2.addLayout(self.verticalLayout_4)
        spacerItem12 = QtWidgets.QSpacerItem(160, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem12)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.predict = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        self.predict.setMinimumSize(QtCore.QSize(150, 0))
        self.predict.setObjectName("predict")
        self.verticalLayout_5.addWidget(self.predict)
        spacerItem13 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem13)
        self.horizontalLayout_2.addLayout(self.verticalLayout_5)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_2)
        spacerItem14 = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem14)
        self.verticalLayout_7.addLayout(self.horizontalLayout_3)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(30, 390, 331, 151))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.comboBox_1 = QtWidgets.QComboBox(self.verticalLayoutWidget_2)
        self.comboBox_1.setMinimumSize(QtCore.QSize(120, 0))
        self.comboBox_1.setObjectName("comboBox_1")
        self.horizontalLayout_5.addWidget(self.comboBox_1)
        self.val_min_1 = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.val_min_1.setMaximumSize(QtCore.QSize(30, 30))
        self.val_min_1.setObjectName("val_min_1")
        self.horizontalLayout_5.addWidget(self.val_min_1)
        self.slider_1 = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        self.slider_1.setOrientation(QtCore.Qt.Horizontal)
        self.slider_1.setObjectName("slider_1")
        self.horizontalLayout_5.addWidget(self.slider_1)
        self.val_max_1 = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.val_max_1.setMaximumSize(QtCore.QSize(30, 30))
        self.val_max_1.setObjectName("val_max_1")
        self.horizontalLayout_5.addWidget(self.val_max_1)
        self.verticalLayout_8.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.comboBox_2 = QtWidgets.QComboBox(self.verticalLayoutWidget_2)
        self.comboBox_2.setMinimumSize(QtCore.QSize(120, 0))
        self.comboBox_2.setObjectName("comboBox_2")
        self.horizontalLayout_6.addWidget(self.comboBox_2)
        self.val_min_2 = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.val_min_2.setMaximumSize(QtCore.QSize(30, 30))
        self.val_min_2.setObjectName("val_min_2")
        self.horizontalLayout_6.addWidget(self.val_min_2)
        self.slider_2 = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        self.slider_2.setOrientation(QtCore.Qt.Horizontal)
        self.slider_2.setObjectName("slider_2")
        self.horizontalLayout_6.addWidget(self.slider_2)
        self.val_max_2 = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.val_max_2.setMaximumSize(QtCore.QSize(30, 30))
        self.val_max_2.setObjectName("val_max_2")
        self.horizontalLayout_6.addWidget(self.val_max_2)
        self.verticalLayout_8.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.comboBox_3 = QtWidgets.QComboBox(self.verticalLayoutWidget_2)
        self.comboBox_3.setMinimumSize(QtCore.QSize(120, 0))
        self.comboBox_3.setObjectName("comboBox_3")
        self.horizontalLayout_7.addWidget(self.comboBox_3)
        self.val_min_3 = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.val_min_3.setMaximumSize(QtCore.QSize(30, 30))
        self.val_min_3.setObjectName("val_min_3")
        self.horizontalLayout_7.addWidget(self.val_min_3)
        self.slider_3 = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        self.slider_3.setOrientation(QtCore.Qt.Horizontal)
        self.slider_3.setObjectName("slider_3")
        self.horizontalLayout_7.addWidget(self.slider_3)
        self.val_max_3 = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.val_max_3.setMaximumSize(QtCore.QSize(30, 30))
        self.val_max_3.setObjectName("val_max_3")
        self.horizontalLayout_7.addWidget(self.val_max_3)
        self.verticalLayout_8.addLayout(self.horizontalLayout_7)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(Dialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.load.setText(_translate("Dialog", "Load"))
        self.save.setText(_translate("Dialog", "Save cfg"))
        self.back.setText(_translate("Dialog", "Back"))
        self.next.setText(_translate("Dialog", "Next"))
        self.predict.setText(_translate("Dialog", "test"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
import tkinter as tk
import tkinter.filedialog
import pandas as pd
import os
from pandastable import Table, TableModel
import pickle
from PIL import ImageTk, Image
from tkinter import ttk
from train import Train
from predict import Predict


class Frontend:
    def __init__(self):
        self.df_data = pd.DataFrame()
        self.filename_data = ''
        self.df_data_predict = pd.DataFrame()
        self.filename_data_predict = ''
        self.desired_columns = []
        self.filename_model = ''
        self.window = tk.Tk()
        self.window.geometry("1800x850")
        self.window.title('Get Variables for Predecting Biological Clocks')
        tk.Label(self.window, text="    TRAIN", font='Helvetica 15').pack(pady=20, side="top", anchor="w")

        # Create a Text Box
        self.label_sheet_name = tk.Label(self.window, text="    Name of the sheet")
        self.label_sheet_name.pack(pady=0, side="top", anchor="w")
        self.sheet_name = tk.Text(self.window, width=20, height=1)
        self.sheet_name.insert(tk.INSERT, "1_Var_CatYNum")
        self.sheet_name.pack(pady=20, side="top", anchor="w")

        self.button_excel_file = tk.Button(self.window, text='Select Excel File with Data',
                                           command=lambda: self.upload_action_excel_file(
                                               self.retrieve_input(self.sheet_name)))
        self.button_excel_file.pack(pady=20, side="top", anchor="w")

        self.label_data_loaded = tk.Label(self.window, text="")
        self.label_data_loaded.pack(pady=0, side="top", anchor="w")

        self.opcion_desired_columns = tk.IntVar()
        genwin = tk.LabelFrame(self.window, text='Select if you want to choose columns')
        genwin.pack(pady=20, side="top", anchor="w")
        self.opcion_desired_columns.set(1)

        tk.Radiobutton(genwin, text='Every columns', variable=self.opcion_desired_columns, value=1).pack(side=tk.TOP)
        tk.Radiobutton(genwin, text='Columns selected with order', variable=self.opcion_desired_columns,
                       value=2).pack(side=tk.TOP)
        tk.Radiobutton(genwin, text='Columns selected with name', variable=self.opcion_desired_columns,
                       value=3).pack(side=tk.TOP)

        self.button_columns_file = tk.Button(genwin, text='Select Txt File with Columns',
                                        command=lambda: self.upload_action_desired_columns())
        self.button_columns_file.pack(pady=20, side="top", anchor="w")
        self.columns_desired_loaded = tk.Label(genwin, text="")
        self.columns_desired_loaded.pack(pady=0, side="top", anchor="w")
        self.desired_columns = []

        self.label_name = tk.Label(self.window, text="Type the label name")
        self.label_name.pack(pady=0, side="top", anchor="w")
        self.label_text = tk.Text(self.window, width=20, height=1)
        self.label_text.insert(tk.INSERT, "DNAmGrimAge")
        self.label_text.pack(pady=20, side="top", anchor="w")

        self.id_name = tk.Label(self.window, text="Type the id name")
        self.id_name.pack(pady=0, side="top", anchor="w")
        self.id_text = tk.Text(self.window, width=20, height=1)
        self.id_text.insert(tk.INSERT, "ID_Muestra")
        self.id_text.pack(pady=20, side="top", anchor="w")

        self.ids_test = []

        self.check_random_test = tk.IntVar()
        frame_upload_ids = tk.Frame(self.window).pack(side=tk.TOP)
        tk.Label(frame_upload_ids, text="If you want specific test, you can upload txt \n "
                             "with ids test, else you can train with random test").pack(anchor="w")

        self.check_random_test_button = tk.Checkbutton(frame_upload_ids, text="Random", variable=self.check_random_test,
                                                       onvalue=1, offvalue=0)
        self.check_random_test_button.pack(anchor="w")
        self.button_ids_file = tk.Button(self.window, text='Select Txt File with ids for testing',
                                         command=lambda: self.upload_action_ids_test(), height=15, width=25)
        self.button_ids_file.pack(pady=20, side="top", anchor="w")

        separator1 = ttk.Separator(self.window, orient='vertical')
        separator1.place(relx=0.185, rely=0, relwidth=0.2, relheight=1)

        parameters_label = tk.Label(self.window, text="PARAMETERS", font='Helvetica 15', padx=10, pady=10)
        parameters_label.place(x=360, y=10)

        self.threshold_variance_value = tk.DoubleVar()
        self.threshold_variance_value.set(0.05)
        self.thresh_variance = tk.LabelFrame(self.window, text='Threshold variance')
        self.thresh_variance.pack(pady=20, side="top", anchor="w")
        self.thresh_variance.place(x=360, y=60)
        tk.Radiobutton(self.thresh_variance, text='0.01', variable=self.threshold_variance_value,
                       value=0.01).pack(side=tk.TOP)
        tk.Radiobutton(self.thresh_variance, text='0.05', variable=self.threshold_variance_value,
                       value=0.05).pack(side=tk.TOP)
        tk.Radiobutton(self.thresh_variance, text='0.07', variable=self.threshold_variance_value,
                       value=0.07).pack(side=tk.TOP)

        self.threshold_importance_value = tk.IntVar()
        self.threshold_importance_value.set(20)
        self.thresh_importance = tk.LabelFrame(self.window, text='Threshold importance')
        self.thresh_importance.pack(pady=20, side="top", anchor="w")
        self.thresh_importance.place(x=360, y=170)
        tk.Radiobutton(self.thresh_importance, text='20', variable=self.threshold_importance_value,
                       value=20).pack(side=tk.TOP)
        tk.Radiobutton(self.thresh_importance, text='30', variable=self.threshold_importance_value,
                       value=30).pack(side=tk.TOP)
        tk.Radiobutton(self.thresh_importance, text='50', variable=self.threshold_importance_value,
                       value=50).pack(side=tk.TOP)

        self.algorithm_imput_value = tk.StringVar()
        self.algorithm_imput_value.set('knn')
        self.algorithm_imput = tk.LabelFrame(self.window, text='Algorithm imput na values')
        self.algorithm_imput.pack(pady=20, side="top", anchor="w")
        self.algorithm_imput.place(x=360, y=280)
        tk.Radiobutton(self.algorithm_imput, text='MEAN', variable=self.algorithm_imput_value,
                       value='mean_mode').pack(side=tk.TOP)
        tk.Radiobutton(self.algorithm_imput, text='KNN', variable=self.algorithm_imput_value,
                       value='knn').pack(side=tk.TOP)
        tk.Radiobutton(self.algorithm_imput, text='LINEAR', variable=self.algorithm_imput_value,
                       value='linear').pack(side=tk.TOP)
        tk.Radiobutton(self.algorithm_imput, text='SVM', variable=self.algorithm_imput_value,
                       value='svm').pack(side=tk.TOP)
        tk.Radiobutton(self.algorithm_imput, text='XGBOOST', variable=self.algorithm_imput_value,
                       value='xgboost').pack(side=tk.TOP)
        tk.Radiobutton(self.algorithm_imput, text='ALL', variable=self.algorithm_imput_value,
                       value='ensemble').pack(side=tk.TOP)

        self.pca_algorithm_value = tk.BooleanVar()
        self.pca_algorithm_value.set(True)
        self.pca_algorithm = tk.LabelFrame(self.window, text='PCA Algorithm')
        self.pca_algorithm.pack(pady=20, side="top", anchor="w")
        self.pca_algorithm.place(x=360, y=450)
        tk.Radiobutton(self.pca_algorithm, text='FALSE', variable=self.pca_algorithm_value,
                       value=False).pack(side=tk.TOP)
        tk.Radiobutton(self.pca_algorithm, text='TRUE', variable=self.pca_algorithm_value,
                       value=True).pack(side=tk.TOP)

        self.ncomp_pca_value = tk.IntVar()
        self.ncomp_pca_value.set(10)
        self.ncomp_pca = tk.LabelFrame(self.window, text='Nº components pca algorithm')
        self.ncomp_pca.pack(pady=20, side="top", anchor="w")
        self.ncomp_pca.place(x=360, y=530)
        tk.Radiobutton(self.ncomp_pca, text='5', variable=self.ncomp_pca_value,
                       value=5).pack(side=tk.TOP)
        tk.Radiobutton(self.ncomp_pca, text='10', variable=self.ncomp_pca_value,
                       value=10).pack(side=tk.TOP)
        tk.Radiobutton(self.ncomp_pca, text='20', variable=self.ncomp_pca_value,
                       value=20).pack(side=tk.TOP)
        tk.Radiobutton(self.ncomp_pca, text='50', variable=self.ncomp_pca_value,
                       value=50).pack(side=tk.TOP)

        self.algorithm_supervised_value = tk.StringVar()
        self.algorithm_supervised_value.set('Linear')
        self.algorithm_supervised = tk.LabelFrame(self.window, text='Algorithm supervised to predict')
        self.algorithm_supervised.pack(pady=20, side="top", anchor="w")
        self.algorithm_supervised.place(x=360, y=650)
        tk.Radiobutton(self.algorithm_supervised, text='LINEAR', variable=self.algorithm_supervised_value,
                       value='Linear').pack(side=tk.TOP)
        tk.Radiobutton(self.algorithm_supervised, text='XGBOOST', variable=self.algorithm_supervised_value,
                       value='XGBoost').pack(side=tk.TOP)
        tk.Radiobutton(self.algorithm_supervised, text='ALL', variable=self.algorithm_supervised_value,
                       value='Ensemble').pack(side=tk.TOP)

        tk.Button(self.window, text='Train Individual', command=self.train_single_model, width=12).place(x=360, y=750)

        tk.Button(self.window, text='Train All Parameters', command=self.train_all_parameters, width=16).place(x=500, y=750)

        separator2 = ttk.Separator(self.window, orient='vertical')
        separator2.place(relx=0.37, rely=0, relwidth=0.2, relheight=1)

        self.parameters = {
            'algorithm_imput': self.algorithm_imput_value.get(),
            'threshold_variance': self.threshold_variance_value.get(),
            'threshold_importance': self.threshold_importance_value.get(),
            'algorithm_supervised': self.algorithm_supervised_value.get(),
            'activated_pca': self.pca_algorithm_value.get(),
            'n_components_pca': self.ncomp_pca_value.get()
        }
        separator3 = ttk.Separator(self.window, orient='vertical')
        separator3.place(relx=0.7, rely=0, relwidth=0.2, relheight=1)

        self.label_sheet_name_predict = tk.Label(self.window, text="Name of the sheet")
        self.label_sheet_name_predict.pack(pady=0, side="top", anchor="w")
        self.label_sheet_name_predict.place(x=1280, y=50)
        self.sheet_name_predict = tk.Text(self.window, width=20, height=1)
        self.sheet_name_predict.insert(tk.INSERT, "")
        self.sheet_name_predict.pack(pady=20, side="top", anchor="w")
        self.sheet_name_predict.place(x=1280, y=85)

        self.button_excel_file_predict = tk.Button(self.window, text='Select Excel File with Data for predicting',
                                                   command=lambda: self.upload_action_excel_file_predict(
                                                    self.retrieve_input(self.sheet_name_predict)))
        self.button_excel_file_predict.pack(pady=20, side="top", anchor="w")
        self.button_excel_file_predict.place(x=1280, y=120)

        self.id_name_predict = tk.Label(self.window, text="Type the id name")
        self.id_name_predict.pack(pady=0, side="top", anchor="w")
        self.id_name_predict.place(x=1280, y=160)
        self.id_text_predict = tk.Text(self.window, width=20, height=1)
        self.id_text_predict.insert(tk.INSERT, "ID_Muestra")
        self.id_text_predict.pack(pady=20, side="top", anchor="w")
        self.id_text_predict.place(x=1280, y=190)

        self.target_name_predict = tk.Label(self.window, text="Type the id name")
        self.target_name_predict.pack(pady=0, side="top", anchor="w")
        self.target_name_predict.place(x=1280, y=220)
        self.target_text_predict = tk.Text(self.window, width=20, height=1)
        self.target_text_predict.insert(tk.INSERT, "DNAmGrimAge")
        self.target_text_predict.pack(pady=20, side="top", anchor="w")
        self.target_text_predict.place(x=1280, y=250)

        self.button_model_predict = tk.Button(self.window, text='Select Model', command=lambda: self.upload_model())
        self.button_model_predict.pack(pady=20, side="top", anchor="w")
        self.button_model_predict.place(x=1280, y=300)

        self.button_predict = tk.Button(self.window, text='PREDICT', command=lambda: self.predict_dataset())
        self.button_predict.pack(pady=20, side="top", anchor="w")
        self.button_predict.place(x=1280, y=340)

    def upload_action_excel_file(self, sheet):
        if len(str(sheet)) > 0:
            try:
                filename = tkinter.filedialog.askopenfilename(initialdir="Data", title='Open files')
                xl_file = pd.ExcelFile(filename)
                dfs = {sheet_name_upload_action: xl_file.parse(sheet_name_upload_action) for sheet_name_upload_action in xl_file.sheet_names}
                df = dfs[sheet]
                df = df.iloc[:60, :]
                self.df_data = df.copy()
                self.filename_data = filename.split(os.sep)[-1]
                self.label_data_loaded.config(text=self.filename_data + ' is loaded correctly')
            except:
                tk.messagebox.showerror('Excel file Error', 'Error: ' + str(sheet) + ' is not a valid sheet')
        else:
            tk.messagebox.showerror('Excel file Error', 'Error: ' + str(sheet) + ' is not a valid sheet')

    def upload_action_excel_file_predict(self, sheet):
        if len(str(sheet)) > 0:
            try:
                filename = tkinter.filedialog.askopenfilename(initialdir="Data", title='Open files')
                xl_file = pd.ExcelFile(filename)
                dfs = {sheet_name_upload_action: xl_file.parse(sheet_name_upload_action)
                       for sheet_name_upload_action in xl_file.sheet_names}
                df = dfs[sheet]
                self.df_data_predict = df.copy()
                self.filename_data_predict = filename.split(os.sep)[-1]
            except:
                tk.messagebox.showerror('Excel file Error', 'Error: ' + str(sheet) + ' is not a valid sheet')
        else:
            tk.messagebox.showerror('Excel file Error', 'Error: ' + str(sheet) + ' is not a valid sheet')

    def upload_action_desired_columns(self):
            try:
                filename = tkinter.filedialog.askopenfilename(initialdir="Data", title='Open files')
                with open(filename) as f:
                    readed_columns = [line.replace('\n', '') for line in f.readlines()]
            except:
                tk.messagebox.showerror('Text file Error', 'Error: It is not available')

            if self.opcion_desired_columns.get() == 2:
                try:
                    columns_with_orger = [int(col) for col in readed_columns]
                    self.desired_columns = [self.df_data.columns.values.tolist()[col] for col in columns_with_orger]
                except:
                    tk.messagebox.showerror('No numeric columns', 'There are some columns no numeric')
            elif self.opcion_desired_columns.get() == 3:
                try:
                    self.desired_columns = [column_name for column_name in readed_columns]

                except:
                    tk.messagebox.showerror('No numeric columns', 'There are some columns no numeric')

    def upload_action_ids_test(self):
            try:
                filename = tkinter.filedialog.askopenfilename(initialdir="Data", title='Open files')
                with open(filename) as f:
                    readed_ids = [line.replace('\n', '') for line in f.readlines()]
            except:
                tk.messagebox.showerror('Text file Error', 'Error: It is not available')
            try:
                self.ids_test = [int(col) for col in readed_ids]
            except:
                tk.messagebox.showerror('No numeric columns', 'There are some columns no numeric')

    def upload_model(self):
            try:
                self.filename_model = tkinter.filedialog.askopenfilename(initialdir="model_store", title='Open files')
            except:
                tk.messagebox.showerror('Text file Error', 'Error: It is not available')

    def retrieve_input(self, my_text_box):
        input = my_text_box.get("1.0", tk.END)
        return input.replace('\n', '')

    def selec(self):
        self.monitor.config(text="Opción {}".format(self.opcion_desired_columns.get()))

    def train_single_model(self):
        col_name_ids = self.retrieve_input(self.id_text)
        label_name = self.retrieve_input(self.label_text)
        id_column = self.retrieve_input(self.id_text)
        self.parameters = {
            'algorithm_imput': self.algorithm_imput_value.get(),
            'threshold_variance': self.threshold_variance_value.get(),
            'threshold_importance': self.threshold_importance_value.get(),
            'algorithm_supervised': self.algorithm_supervised_value.get(),
            'activated_pca': self.pca_algorithm_value.get(),
            'n_components_pca': self.ncomp_pca_value.get()
        }
        if self.check_random_test.get() == 0 and len(self.ids_test) == 0:
            tk.messagebox.showerror('Ids Error', 'You have to select random ids or upload your own ids')
        elif self.check_random_test.get() == 1:
            df_aux = self.df_data.sample(frac=0.20, axis='rows')

            ids_test = list(df_aux[col_name_ids])
            if len(self.desired_columns) > 0:
                list_columns = self.desired_columns
            else:
                list_columns = df_aux.columns.values.tolist()

            train_object = Train(list_columns=list_columns,
                                 df_data=self.df_data,
                                 label_name=label_name,
                                 id_column=id_column,
                                 parameters=self.parameters,
                                 individual_all='Individual',
                                 ids_test=ids_test)
            rmse, mae, r2, df, best_5_features = train_object.predict()
            tk.messagebox.showinfo(title="Train finished", message="Train is finished successfully")
            image1 = Image.open(os.path.join('Results', label_name + '_graphics.png'))
            image1 = image1.resize((480, 360), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(image1)

            image_display = tkinter.Label(image=img)
            image_display.image = img
            image_display.place(x=680, y=20)

            label_features = tk.Label(self.window, text="The main features are: \n" + str(best_5_features[0]) +
                                      '\n' + str(best_5_features[1]) + '\n' + str(best_5_features[2]))
            label_features.pack(pady=0, side="top", anchor="w")
            label_features.place(x=680, y=390)

            label_metric = tk.Label(self.window, text="RMSE: " + str(round(rmse, 3)) + "\tMAE: " + str(round(mae, 3))
                                    + "\tR2: " + str(round(r2, 3)))
            label_metric.pack(pady=0, side="top", anchor="w")
            label_metric.place(x=680, y=470)

            frame_table_train = tk.Frame(self.window)
            frame_table_train.pack(pady=0, side="top", anchor="w")
            frame_table_train.place(x=680, y=500)
            table_train = Table(frame_table_train, showtoolbar=True, showstatusbar=True)
            table_train.importCSV(os.path.join('Results', label_name + '_PredictedVsTrue.csv'))
            table_train.autoResizeColumns()
            table_train.show()

        else:
            if len(self.desired_columns) > 0:
                list_columns = self.desired_columns
            else:
                list_columns = self.df_data.columns.values.tolist()
            train_object = Train(list_columns=list_columns,
                                 df_data=self.df_data,
                                 label_name=label_name,
                                 id_column=id_column,
                                 parameters=self.parameters,
                                 individual_all='Individual',
                                 ids_test=self.ids_test)
            rmse, mae, r2, df, best_5_features = train_object.predict()

            image1 = Image.open(os.path.join('Results', label_name + '_graphics.png'))
            image1 = image1.resize((480, 360), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(image1)

            image_display = tkinter.Label(image=img)
            image_display.image = img
            image_display.place(x=680, y=20)

            label_features = tk.Label(self.window, text="The main features are: \n" + str(best_5_features[0]) +
                                                        '\n' + str(best_5_features[1]) + '\n' + str(best_5_features[2]))
            label_features.pack(pady=0, side="top", anchor="w")
            label_features.place(x=680, y=390)
            label_metric = tk.Label(self.window, text="RMSE: " + str(round(rmse, 3)) + "\tMAE: " + str(round(mae, 3))
                                    + "\tR2: " + str(round(r2, 3)))
            label_metric.pack(pady=0, side="top", anchor="w")
            label_metric.place(x=680, y=470)

            frame_table_train_all = tk.Frame(self.window)
            frame_table_train_all.pack(pady=0, side="top", anchor="w")
            frame_table_train_all.place(x=680, y=500)
            table_train_all = Table(frame_table_train_all, showtoolbar=True, showstatusbar=True)
            table_train_all.importCSV(os.path.join('Results', label_name + '_PredictedVsTrue.csv'))
            table_train_all.autoResizeColumns()
            table_train_all.show()
            tk.messagebox.showinfo(title="Train finished", message="Train is finished successfully")

    def train_all_parameters(self):
        col_name_ids = self.retrieve_input(self.id_text)
        label_name = self.retrieve_input(self.label_text)
        id_column = self.retrieve_input(self.id_text)
        self.parameters = {
            'algorithm_imput': self.algorithm_imput_value.get(),
            'threshold_variance': self.threshold_variance_value.get(),
            'threshold_importance': self.threshold_importance_value.get(),
            'algorithm_supervised': self.algorithm_supervised_value.get(),
            'activated_pca': self.pca_algorithm_value.get(),
            'n_components_pca': self.ncomp_pca_value.get()
        }
        if self.check_random_test.get() == 0 and len(self.ids_test) == 0:
            tk.messagebox.showerror('Ids Error', 'You have to select random ids or upload your own ids')
        elif self.check_random_test.get() == 1:
            df_aux = self.df_data.sample(frac=0.20, axis='rows')

            ids_test = list(df_aux[col_name_ids])
            if len(self.desired_columns) > 0:
                list_columns = self.desired_columns
            else:
                list_columns = df_aux.columns.values.tolist()

            train_object = Train(list_columns=list_columns,
                                 df_data=self.df_data,
                                 label_name=label_name,
                                 id_column=id_column,
                                 parameters=self.parameters,
                                 individual_all='All',
                                 ids_test=ids_test)
            rmse, mae, r2, df, best_5_features = train_object.predict()
            tk.messagebox.showinfo(title="Train finished", message="Train is finished successfully")
            image1 = Image.open(os.path.join('Results', label_name + '_graphics.png'))
            image1 = image1.resize((480, 360), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(image1)

            image_display = tkinter.Label(image=img)
            image_display.image = img
            image_display.place(x=680, y=20)

            label_features = tk.Label(self.window, text="The main features are: \n" + str(best_5_features[0]) +
                                                        '\n' + str(best_5_features[1]) + '\n' + str(best_5_features[2]))
            label_features.pack(pady=0, side="top", anchor="w")
            label_features.place(x=680, y=390)

            label_metric = tk.Label(self.window, text="RMSE: " + str(round(rmse, 3)) + "\tMAE: " + str(round(mae, 3))
                                    + "\tR2: " + str(round(r2, 3)))
            label_metric.pack(pady=0, side="top", anchor="w")
            label_metric.place(x=680, y=470)

            frame = tk.Frame(self.window)
            frame.pack(pady=0, side="top", anchor="w")
            frame.place(x=680, y=500)
            table = Table(frame, showtoolbar=True, showstatusbar=True)
            table.importCSV(os.path.join('Results', label_name + '_PredictedVsTrue.csv'))
            table.show()
        else:
            if len(self.desired_columns) > 0:
                list_columns = self.desired_columns
            else:
                list_columns = self.df_data.columns.values.tolist()
            train_object = Train(list_columns=list_columns,
                                 df_data=self.df_data,
                                 label_name=label_name,
                                 id_column=id_column,
                                 parameters=self.parameters,
                                 individual_all='All',
                                 ids_test=self.ids_test)
            rmse, mae, r2, df, best_5_features = train_object.predict()

            image1 = Image.open(os.path.join('Results', label_name + '_graphics.png'))
            image1 = image1.resize((480, 360), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(image1)

            image_display = tkinter.Label(image=img)
            image_display.image = img
            image_display.place(x=680, y=20)

            label_features = tk.Label(self.window, text="The main features are: \n" + str(best_5_features[0]) +
                                                        '\n' + str(best_5_features[1]) + '\n' + str(best_5_features[2]))
            label_features.pack(pady=0, side="top", anchor="w")
            label_features.place(x=680, y=390)

            label_metric = tk.Label(self.window, text="RMSE: " + str(round(rmse, 3)) + "\tMAE: " + str(round(mae, 3))
                                    + "\tR2: " + str(round(r2, 3)))
            label_metric.pack(pady=0, side="top", anchor="w")
            label_metric.place(x=680, y=470)

            frame = tk.Frame(self.window)
            frame.pack(pady=0, side="top", anchor="w")
            frame.place(x=680, y=500)
            table = Table(frame, showtoolbar=True, showstatusbar=True)
            table.importCSV(os.path.join('Results', label_name + '_PredictedVsTrue.csv'))
            table.show()
            tk.messagebox.showinfo(title="Train finished", message="Train is finished successfully")

    def predict_dataset(self):
        id_column = self.retrieve_input(self.id_text_predict)
        target = self.retrieve_input(self.target_text_predict)
        predict_object = Predict(self.df_data_predict, id_column, target, self.filename_model)
        predict_object.predict()
        frame_table_predict = tk.Frame(self.window)
        frame_table_predict.pack(pady=0, side="top", anchor="w")
        frame_table_predict.place(x=1280, y=500)
        table_predict = Table(frame_table_predict, showtoolbar=True, showstatusbar=True)
        table_predict.importCSV(os.path.join('Results', target + '_PredictedResults.csv'))
        table_predict.autoResizeColumns()
        table_predict.show()
        tk.messagebox.showinfo(title="Predict finished", message="Predict is finished successfully")

    def start(self) -> None:
        self.window.mainloop()


if __name__ == '__main__':
    app = Frontend()
    app.start()

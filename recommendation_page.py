import streamlit as st
from copy import deepcopy
from model_GGNN import *
from utils import *
from rerank import *


def show_recommendation_page():
    st.write("# Recommendation Steps:")

    dataset = r'https://raw.githubusercontent.com/AI-N/fair-srs-app/main/data/sample.csv'
    df = pd.read_csv(dataset, delimiter=',')

    user = st.slider('user', min_value=0, max_value=df['user'].max(), value=67)  # this is a widget
    # user = st.selectbox("select a user", ("user 1", "user 2", "user 3"))
    st.write('The selected user_id: ', user)

    df_user = df[df['user'] == user]

    # item_clicks
    #print('finding source and targets of clicks for deepwalk: it may take long time')
    #item_clicks = pd.DataFrame(columns=['source', 'target'])
    #for i in df['session_id'].unique():
    #    lenght = len(df[df['session_id'] == i])
    #    for j in range(lenght - 1):
    #        item_clicks = item_clicks.append({'source': df[df['session_id'] == i].reset_index(drop=True)['item'][j],
    #                                          'target': df[df['session_id'] == i].reset_index(drop=True)['item'][
    #                                              j + 1]}, ignore_index=True)
    #item_clicks.drop_duplicates(keep="first", inplace=True)
    #item_clicks = item_clicks.reset_index(drop=True)
    item_clicks = pd.read_csv(r'https://raw.githubusercontent.com/AI-N/fair-srs-app/main/data/item_clicks.csv', delimiter=',')

    # Split out %20 of each user's sessions as test set
    df = df.drop(['t'], axis=1)
    test_size = 0.3
    test = []
    train = []
    for i in df['user'].unique():
        dd = df[df['user'] == i]
        minimum = dd['session_id'].min()
        if minimum == 0:
            minimum = minimum - 1
        maximum = dd['session_id'].max()
        lenght = maximum - minimum + 1
        splitpoint = int(lenght * test_size)
        if minimum == 0:
            for j in range(maximum - minimum - splitpoint):
                train.append(dd[dd['session_id'] == minimum + j].values.tolist())
            for j in range(splitpoint):
                test.append(dd[dd['session_id'] == maximum - j].values.tolist())
        else:
            for j in range(maximum - minimum - splitpoint + 1):
                train.append(dd[dd['session_id'] == minimum + j].values.tolist())
            for j in range(splitpoint):
                test.append(dd[dd['session_id'] == maximum - j].values.tolist())
    tt = [item for sublist in train for item in sublist]
    tt = pd.DataFrame.from_records(tt)
    trainn = tt.rename(columns={0: "ts", 1: "item", 2: "session_id", 3: "user"})
    tt = [item for sublist in test for item in sublist]
    tt = pd.DataFrame.from_records(tt)
    testt = tt.rename(columns={0: "ts", 1: "item", 2: "session_id", 3: "user"})


    ## Convert test and train sessions to sequences and labels
    def Seq_without_uid(data, df, train=True):
        if train:
            df = pd.DataFrame(columns=['session_id', 'seq'])
        else:
            df = pd.DataFrame(columns=['session_id', 'seq', 'next_click'])
        sid = data['session_id'].unique()
        for en, i in enumerate(sid):
            s = data[data['session_id'] == i]
            if train:
                to_append = pd.Series([i, s['item_id'].tolist()]).tolist()
            else:
                to_append = pd.Series([i, s['item_id'][0:-1].tolist(), s['item_id'].tolist()[-1]]).tolist()
            a_series = pd.Series(to_append, index=df.columns)
            df = df.append(a_series, ignore_index=True)
        return df

    def Seq_with_uid(data, df, train=True):
        if train:
            df = pd.DataFrame(columns=['user_id', 'session_id', 'seq'])
        else:
            df = pd.DataFrame(columns=['user_id', 'session_id', 'seq', 'next_click'])
        uid = data['user'].unique()
        for en, i in enumerate(uid):
            # if en%3000==0:
            # print(en)
            u = data[data['user'] == i]
            s = u['session_id'].unique()
            for j in s:
                u_s = u[u['session_id'] == j]
                if train:
                    to_append = pd.Series([i, j, u_s['item'].tolist()]).tolist()
                else:
                    to_append = pd.Series([i, j, u_s['item'][0:-1].tolist(), u_s['item'].tolist()[-1]]).tolist()
                a_series = pd.Series(to_append, index=df.columns)
                df = df.append(a_series, ignore_index=True)
        return df

    df_user_t = Seq_with_uid(testt[testt['user'] == user], df_user, train=False)
    df_user_tr = Seq_with_uid(trainn[trainn['user'] == user], df_user, train=True)
    # df_user_t
    df_user_c = pd.concat([df_user_tr, df_user_t], ignore_index=True)
    df_user_c = df_user_c.sort_values(by=['session_id'])
    df_user_c = df_user_c.reset_index(drop=True)
    # df_user_c
    # st.write(df_user_t)
    # st.write("train sessions for ", user, " :")
    # st.write(df_user_tr)

    df_user_t = testt[testt['user'] == user]
    df_user_tr = trainn[trainn['user'] == user]
    df_user = df[df['user'] == user]


    def user_item_click(data):
        item_cl = pd.DataFrame(columns=['source', 'target', 'count'])
        for i in data['session_id'].unique():
            lenght = len(data[data['session_id'] == i])
            for j in range(lenght - 1):
                item_cl = item_cl.append({'source': data[data['session_id'] == i].reset_index(drop=True)['item'][j],
                                          'target': data[data['session_id'] == i].reset_index(drop=True)['item'][j + 1],
                                          'count': 0}, ignore_index=True)
        dups_user_t = item_cl.pivot_table(index=['source', 'target'], aggfunc='size')
        for i in range(len(item_cl)):
            a = item_cl['source'][i]
            b = item_cl['target'][i]
            item_cl.loc[(item_cl['source'] == a) & (item_cl['target'] == b), 'count'] = dups_user_t.loc[a].loc[b]
        item_cl.drop_duplicates(keep="first", inplace=True)
        item_cl = item_cl.reset_index(drop=True)
        return item_cl

    user_item_clicks = user_item_click(df_user)
    # user_item_clicks_t = user_item_click(df_user_t)
    # user_item_clicks_tr = user_item_click(df_user_tr)


    st.write("Please wait for few seconds to build the test and train data....")
    df_t = Seq_with_uid(testt, df, train=False)
    df_tr = Seq_with_uid(trainn, df, train=True)
    te_ids = df_t['session_id']
    te_seqs = df_t['seq']
    te_labs = df_t['next_click']

    def process_seqs(iseqs):
        out_seqs = []
        labs = []
        ids = []
        for id, seq in zip(range(len(iseqs)), iseqs):
            for i in range(1, len(seq)):
                tar = seq[-i]
                labs += [tar]
                out_seqs += [seq[:-i]]
                ids += [id]
        return out_seqs, labs, ids

    def process_seqs_tes(iseqs, ilabs):
        out_seqs = []
        labs = []
        ids = []
        for id, seq in zip(range(len(iseqs)), iseqs):
            out_seqs += [seq]
            ids += [id]
        for id, lab in zip(range(len(ilabs)), ilabs):
            labs += [lab]
        return out_seqs, labs, ids

    te_seqs, te_labs, te_ids = process_seqs_tes(df_t['seq'], df_t['next_click'])
    tr_seqs, tr_labs, tr_ids = process_seqs(df_tr['seq'])
    tra = (tr_seqs, tr_labs)
    tes = (te_seqs, te_labs)

    def Sess_viz(sess_data1):
        G = nx.Graph()
        sources = sess_data1['source']
        targets = sess_data1['target']
        weights = sess_data1['count']
        edge_data = zip(sources, targets, weights)
        for e in edge_data:
            src = 'item ' + str(e[0])
            dst = 'item ' + str(e[1])
            w = e[2]
            G.add_edge(src, dst, weight=w, title=w)  # , title=w
        dot = nx.nx_pydot.to_pydot(G)
        return st.graphviz_chart(dot.to_string())
    #st.write("### Check the box if you want to see a graph visualization of this user's sessions!")
    #page1 = st.checkbox("user session graph visualization!")
    #if page1:
    #    sess_data = user_item_clicks
    #    st.write("Graph visualization sessions for user ", user)
    #    Sess_viz(sess_data)
    df_current = df_user_c.tail(1)
    df_historical = df_user_c.iloc[:-1, :]
    df_historical = df_historical.drop(['next_click'], axis=1)
    st.write("historical sessions for user ", user, " : ")
    st.write(df_historical)
    st.write("current session for user ", user, " : ")
    st.write(df_current)
    st.write("Click '**Graph Representation**' if you like to see the user", user, " session graphs.")
    start_viz = st.button('Graph Representation')
    if start_viz:
        sess_data = user_item_clicks
        st.write("## Session graph visualization for user ", user)
        Sess_viz(sess_data)

    st.write("Click '**Run Model**' to run Fair-SRS model.")
    start_execution = st.button('Run Model')#Run DeepWalk model
    if start_execution:
        gif_runner = st.image('https://aws1.discourse-cdn.com/business7/uploads/streamlit/original/2X/2/247a8220ebe0d7e99dbbd31a2c227dde7767fbe1.gif')
        #result = run_model(args)
        #gif_runner.empty()
        #display_output(result)
        st.write('Please wait few seconds for **DeepWalk model training**...')
        A = nx.from_pandas_edgelist(item_clicks, "source", "target", edge_attr=None, create_using=nx.Graph())

        # function to generate random walk sequences of nodes
        def get_randomwalk(node, path_length):
            random_walk = [node]
            for i in range(path_length - 1):
                temp = list(A.neighbors(node))
                temp = list(set(temp) - set(random_walk))
                if len(temp) == 0:
                    break
                random_node = random.choice(temp)
                random_walk.append(random_node)
                node = random_node
            return random_walk

        all_nodes = list(A.nodes())
        random_walks = []
        for n in tqdm(all_nodes):
            for i in range(5):
                random_walks.append(get_randomwalk(n, 10))
        # train word2vec model
        model_deepwalk = Word2Vec(window=4, sg=1, hs=0,
                                  negative=10,  # for negative sampling
                                  alpha=0.03, min_alpha=0.0007,
                                  seed=14)
        #st.write('DeepWalk model:', model_deepwalk)
        model_deepwalk.build_vocab(random_walks, progress_per=2)
        model_deepwalk.train(random_walks, total_examples=model_deepwalk.corpus_count, epochs=15, report_delay=1)
        # terms=item_clicks['source'].unique()
        # plot_nodes(terms)
        #gif_runner.empty()
        #st.write("Done")
        df_t['Sim_list'] = ""  # df_t is the dataframe for test sequences (refer to preprocess.py)
        for ind in range(len(df_t)):
            if len(df_t['seq'][ind]) == 1:
                df_t['Sim_list'][ind] = 1
            else:
                tsim = []
                for counter, i in enumerate(df_t['seq'][ind][0:len(df_t['seq'][ind]) - 1]):
                    for j in df_t['seq'][ind][counter + 1:len(df_t['seq'][ind])]:
                        sim = model_deepwalk.wv.similarity(int(i), int(j))  # sim with Word2vec model
                        tsim.append(sim)
                df_t['Sim_list'][ind] = sum(tsim) / len(tsim)

        #st.write('Done!')
        # 2D plot of items in the trained Word2Vev model
        word_list = item_clicks['source'].unique()
        X = model_deepwalk.wv[word_list]
        # reduce dimensions to 2
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        fig = plt.figure(figsize=(14, 14))
        # create a scatter plot of the projection
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(result[:, 0], result[:, 1], alpha=0.3, label='items')
        user_list_t = np.unique(
            df_t[df_t['user_id'] == user]['seq'][df_t[df_t['user_id'] == user]['seq'].index[0]])
        for i, word in enumerate(user_list_t):
            ax.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=10, color='darkred',
                         label='user current session')
            # user_list_tr=df_t['test_seq'][user]
        # for i, word in enumerate(user_list_tr):
        #    plt.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=15, color='black')
        plt.title('user {}'.format(user), fontsize=15)
        plt.legend()
        st.write("### 2D node embedding visualization of items in the trained DeepWalk model:")
        st.write("(**red** annotated nodes are items in user's current session)")
        st.write(fig)

        st.write("## Results from DeepWalk model for user ", user," :")
        st.write("The similarity of items in curresnt session for user ", user, " is: ",
                 round((df_t['Sim_list'][user]) * 100, 4), "%")
        st.write("The Level of Interest to Diversity (LID) in curresnt session for user ", user, " is: ",
                 round((1 - df_t['Sim_list'][user]) * 100, 4), "%")

        st.write('## Prediction Model...')
    #start_pr = st.button('Prediction/Recommendation')
    #if start_pr:
        #gif_runner1 = st.image('https://aws1.discourse-cdn.com/business7/uploads/streamlit/original/2X/2/247a8220ebe0d7e99dbbd31a2c227dde7767fbe1.gif')
        # torch.cuda.set_device(1)
        batchSize = 100  # 'input batch size'
        hiddenSize = 100  # 'hidden state size'
        epoch = 4  # 'the number of epochs to train for'
        lr = 0.001  # 'learning rate'  # [0.001, 0.0005, 0.0001]
        lr_dc = 0.1  # 'learning rate decay rate'
        lr_dc_step = 3  # 'the number of steps after which the learning rate decay'
        l2 = 1e-5  # 'l2 penalty'  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
        step = 1  # 'gnn propogation steps'
        patience = 10  # 'the number of epoch to wait before early stop '
        nonhybrid = False  # 'only use the global preference to predict'
        validation = False  # 'validation'
        valid_portion = 0.1  # 'split the portion of training set as validation set'
        norm = True  # 'adapt NISER, l2 norm over item and session embedding'
        TA = False  # 'use target-aware or not'
        scale = True  # 'scaling factor sigma'
        alpha = 0.5  # alpha: to determine the proportion of long-tail items in top-k recommendation based on user LID'#[0.1, 0.5, 1.0]
        train_data = tra
        if validation:
            train_data, valid_data = split_validation(train_data, valid_portion)
            test_data = valid_data
        else:
            test_data = tes
        train_data = Data(train_data, shuffle=True)
        test_data = Data(test_data, shuffle=False)
        n_node = 5000
        model = trans_to_cuda(SessionGraph(n_node,hiddenSize,norm,TA,scale,batchSize,nonhybrid,step,lr,l2,lr_dc_step,lr_dc))
        # print(model)
        best_result = [0, 0, 0, 0, 0, 0]
        for epoch in range(epoch):
            hit5, mrr5, hit10, mrr10, hit20, mrr20, targets_, scores5_value, scores5_ind, scores10_value, scores10_ind, scores20_value, scores20_ind, scores500_value, scores500_ind = train_test(
                model, train_data, test_data)
            if hit5 >= best_result[0]:
                best_result[0] = hit5
            if mrr5 >= best_result[1]:
                best_result[1] = mrr5
            if hit10 >= best_result[2]:
                best_result[2] = hit10
            if mrr10 >= best_result[3]:
                best_result[3] = mrr10
            if hit20 >= best_result[4]:
                best_result[4] = hit20
            if mrr20 >= best_result[5]:
                best_result[5] = mrr20
        alpha = 0.5
        pop = []
        for i in df['item'].unique():
            pop.append(len(trainn[trainn['item'] == i]) / len(df['item'].unique()))
        pop_max = max(pop)
        pop_df = pd.DataFrame(pop, columns=['pop'])
        pop = pop_df / pop_max
        unpop_ = 1 - (pop_df / pop_max)
        unpop = unpop_.rename(columns={"pop": "unpop"})

        ## Assign set of long-tail items
        ind = unpop[unpop['unpop'] >= 0.995].index  # can be chosen to have 10% of less popular items as long-tail items
        # print(ind)
        lt = df['item'][ind].tolist()
        # print('These two should have same length (if not: something went wrong):',len(LID),len(scores5_ind))
        scores5_ind1 = deepcopy(scores5_ind)
        scores5_ind_new = LT_inclusion(scores5_ind, alpha, df_t, lt, scores500_ind)
        scores10_ind1 = deepcopy(scores10_ind)
        scores10_ind_new = LT_inclusion(scores10_ind, alpha, df_t, lt, scores500_ind)
        scores20_ind1 = deepcopy(scores20_ind)
        scores20_ind_new = LT_inclusion(scores20_ind, alpha, df_t, lt, scores500_ind)

        gif_runner.empty()
        #gif_runner1.empty()

        # top-5:
        be = list(set([item for sublist in scores5_ind1 for item in sublist]))
        af = list(set([item for sublist in scores5_ind_new for item in sublist]))
        st.write("Recall@5:", round(best_result[0], 4), "  MMR@5:", round(best_result[1], 4),
                 "  Cov_unpop@5:", round(sum(pd.Series(af).isin(lt)) / len(lt), 4))

        # top-10:
        be = list(set([item for sublist in scores10_ind1 for item in sublist]))
        af = list(set([item for sublist in scores10_ind_new for item in sublist]))
        st.write("Recall@10:", round(best_result[2], 4), "  MMR@10:", round(best_result[3], 4),
                 "  Cov_unpop@5:", round(sum(pd.Series(af).isin(lt)) / len(lt), 4))
        # top-20:
        be = list(set([item for sublist in scores20_ind1 for item in sublist]))
        af = list(set([item for sublist in scores20_ind_new for item in sublist]))
        st.write("Recall@20:", round(best_result[4], 4), "  MMR@20:", round(best_result[5], 4),
                 "  Cov_unpop@5:", round(sum(pd.Series(af).isin(lt)) / len(lt), 4))

        st.write("## Top-5 recommendations for user",user,":",
                 pd.DataFrame(scores5_ind[user], index =['1', '2', '3', '4', '5'],
                              columns =['Top-5 recommendations']) )



        st.write("""Feel free to choose another user and see the results! You can find the comparison of Fair-SRS with existing works in the paper!""")
        st.write("""Note that the results are different from the paper as in this demo we use a **sub sample of dataset** to make it run faster!""")





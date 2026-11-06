from src_T1.runners.runner_base import Runner_Base


class Runner_UDADS_Net(Runner_Base):

    def forward(self, data, model):
        X, y = data
        p1, _, emb, p2, p3 = model(X)

        return {'p1':p1, 'p2':p2, 'p3':p3, 'emb':emb, 'p_rate':p1, 'y':y}

    def forward_loss(self, res):
        p1, p2, p3, emb, y = res['p1'], res['p2'], res['p3'], res['emb'], res['y']
        GUL_res_1 = self.GUL_func(p1, y[:, [0]], None)
        GUL_res_2 = self.GUL_func(p2, y[:, [0]], None)
        GUL_res_3 = self.GUL_func(p3, y[:, [0]], None)
        BCE_res_1 = self.BCE_func(p1, y[:, [0]], None)
        BCE_res_2 = self.BCE_func(p2, y[:, [0]], None)
        BCE_res_3 = self.BCE_func(p3, y[:, [0]], None)
        loss_dict = {}
        loss_dict['GUL'] = (GUL_res_1 + GUL_res_2 + GUL_res_3) / 2
        loss_dict['BCE'] = (BCE_res_1 + BCE_res_2 + BCE_res_3) / 2
        loss_final = loss_dict['GUL']*self.hp.loss_weight['GUL'] + loss_dict['BCE']*self.hp.loss_weight['BCE']

        return loss_dict, loss_final

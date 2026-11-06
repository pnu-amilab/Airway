from src_T1.runners.runner_base import Runner_Base


class Runner_TSE_Net(Runner_Base):

    def forward(self, data, model):
        X, y = data
        p, _, emb, c = model(X)

        return {'p':p, 'c':c, 'emb':emb, 'p_rate':p, 'y':y}

    def forward_loss(self, res):
        p, c, emb, y = res['p'], res['c'], res['emb'], res['y']
        GUL_res_p = self.GUL_func(p, y[:, [0]], None)
        GUL_res_c = self.GUL_func(c, y[:, [1]], None)
        BCE_res_p = self.BCE_func(p, y[:, [0]], None)
        BCE_res_c = self.BCE_func(c, y[:, [1]], None)
        loss_dict = {}
        loss_dict['GUL'] = (GUL_res_p + GUL_res_c) / 2
        loss_dict['BCE'] = (BCE_res_p + BCE_res_c) / 2
        loss_final = loss_dict['GUL']*self.hp.loss_weight['GUL'] + loss_dict['BCE']*self.hp.loss_weight['BCE']

        return loss_dict, loss_final

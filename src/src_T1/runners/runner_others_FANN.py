from src_T1.runners.runner_base import Runner_Base


class Runner_FANN(Runner_Base):

    def forward(self, data, model):
        X, y = data
        p1, _, emb, p2, p3, p4 = model(X)

        return {'p1':p1, 'p2':p2, 'p3':p3, 'p4':p4, 'emb':emb, 'p_rate':p1, 'y':y}

    def forward_loss(self, res):
        y = res['y']
        ps = [res['p1'], res['p2'], res['p3'], res['p4']]
        loss_dict = {}
        loss_dict['GUL'] = sum([self.GUL_func(p, y[:, [0]], None) for p in ps]) / 4
        loss_dict['BCE'] = sum([self.BCE_func(p, y[:, [0]], None) for p in ps]) / 4
        loss_final = loss_dict['GUL']*self.hp.loss_weight['GUL'] + loss_dict['BCE']*self.hp.loss_weight['BCE']

        return loss_dict, loss_final

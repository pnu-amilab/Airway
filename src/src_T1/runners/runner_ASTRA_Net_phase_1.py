from src_T1.runners.runner_base import Runner_Base


class Runner_ASTRA_Net_phase_1(Runner_Base):

    def __init__(self, hp):
        super().__init__(hp)
        self.loss_keys = ['GUL', 'BCE', 'CTL', 'BGR']

    def forward(self, data, model):
        X, y = data
        p1, p2, emb = model(X)

        return {'p1':p1, 'p2':p2, 'emb':emb, 'p_rate':p2, 'y':y}

    def forward_loss(self, res):
        p1, p2, emb, y = res['p1'], res['p2'], res['emb'], res['y']
        GUL_res_E = self.GUL_func(p1, y[:, [0]], y[:, [3]])
        GUL_res_D = self.GUL_func(p2, y[:, [0]], y[:, [3]])
        BCE_res_E = self.BCE_func(p1, y[:, [0]], y[:, [3]])
        BCE_res_D = self.BCE_func(p2, y[:, [0]], y[:, [3]])
        CTL_res_E = self.CTL_func(p1, y[:, [1]], y[:, [3]])
        CTL_res_D = self.CTL_func(p2, y[:, [1]], y[:, [3]])
        BGR_res = self.BGR_func(emb, y[:, [0]])

        loss_dict = {'GUL':GUL_res_D, 'BCE':BCE_res_D, 'CTL':CTL_res_D, 'BGR':BGR_res}
        loss_GUL = (GUL_res_E+GUL_res_D)/2
        loss_BCE = (BCE_res_E+BCE_res_D)/2
        loss_CTL = (CTL_res_E+CTL_res_D)/2
        loss_final = (loss_GUL*self.hp.loss_weight['GUL']+
                      loss_BCE*self.hp.loss_weight['BCE']+
                      loss_CTL*self.hp.loss_weight['CTL']+
                      BGR_res*self.hp.loss_weight['BGR'])

        return loss_dict, loss_final

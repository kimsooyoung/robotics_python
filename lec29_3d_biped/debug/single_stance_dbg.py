


if __name__=="__main__":
    t0, t1 = 0, 2

    sol2 = solve_ivp(
        single_stance, [t0, t1], z_afs, method='RK45', t_eval=t_span,
        dense_output=True, events=midstance, atol = 1e-13, rtol = 1e-13, 
        args=(params,)
    )
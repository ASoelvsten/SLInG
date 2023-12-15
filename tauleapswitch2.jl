function tauleapswitch2(temp_data)
    temp_X = temp_data.X

    # Rates initialization
    c = zeros(temp_data.M)
    for reaction = 1:temp_data.M
        c[reaction] = rates(temp_data, reaction)
    end
    species_ts = copy(temp_data.X)
    r = [PoissonRandom.pois_rand(v * temp_data.tau) for v in c]

    # update species after reaction
    for reaction = 1:temp_data.M
        if !iszero(r[reaction])                # if is not zero we update the reaction
            for k = 1:4
                ind = temp_data.cm_rea[reaction,k]
                if !iszero(temp_data.cm_rea[reaction,k])
                    temp_X[ind] = temp_data.X[ind] + r[reaction] * temp_data.stoichio[reaction,k]
                end
            end
        end
    end

    if any(temp_X .< 0)
        temp_data.X[1:end] = species_ts
        output_species_ts = ssa_switch4(temp_data)
        return output_species_ts
    else
        return temp_X
    end
end

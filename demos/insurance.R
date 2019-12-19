## Plot insurance data

library(tidyverse)
library(ggthemes)
library(latex2exp)

df <- read_csv("insurance_data.csv")

show_pd <- function(df, a0) {
    df %>%
        filter(k %in% c(0, 0.1, 1, 10, 75)) %>%
        filter(a0 == !!a0) %>%
        ggplot(aes(nd,
                   fill = factor(beta))) +
        geom_histogram(position = "dodge",
                       bins = 75) +
        facet_grid(k ~ lij) +
        scale_fill_colorblind() +
        theme_tufte()
}

show_cost <- function(df, a0) {
    df %>%
        filter(k %in% c(0, 0.1, 1, 10, 75)) %>%
        filter(a0 == !!a0) %>%
        ggplot(aes(nd, cost,
                   color = factor(beta))) +
        geom_point(alpha = 0.4,
                   size = 0.5) +
        facet_grid(k ~ lij) +
        scale_color_colorblind() +
        theme_tufte() +
        theme(panel.border = element_rect(fill = NA,
                                          color = "lightgray")) +
        labs(x = "Defaults",
             y = "Cost",
             color = TeX("$\\beta$"),
             title = TeX(paste0("$a_0 = ", a0, "$")))
}

show_avg_cost <- function(df, nd_min) {
    df %>%
        ggplot(aes(k, cost,
                   color = nd)) +
        geom_jitter(alpha = 0.4) +
        scale_x_log10() +
        scale_color_viridis_c() +
        facet_grid(lij ~ a0 + beta) +
        theme_tufte() +
        theme(panel.border = element_rect(fill = NA,
                                          color = "lightgray"))
    ## df %>%
    ##     filter(nd >= nd_min) %>%
    ##     group_by(N, sigma, k, beta, lij, a0) %>%
    ##     summarize(avg_cost = mean(cost)) %>%
    ##     ungroup() %>%
    ##     ggplot(aes(k, avg_cost,
    ##                color = factor(beta))) +
    ##     geom_line() +
    ##     scale_x_log10() +
    ##     scale_color_colorblind() +
    ##     facet_grid(lij ~ a0) +
    ##     theme_tufte()
}


#ifndef CB_MAILBOX_H
#define CB_MAILBOX_H

#include <stdint.h>

#include "cbconst.h"
#include "board.h"

static inline cb_ptype_t cb_m_at_sq(cb_mailbox_t *mailbox, uint8_t sq)
{
    return mailbox->data[sq];
}

static inline cb_ptype_t cb_m_at(cb_mailbox_t *mailbox, uint8_t row, uint8_t col)
{
    return cb_m_at_sq(mailbox, row * 8 + col);
}

#endif /* CB_MAILBOX_H */
